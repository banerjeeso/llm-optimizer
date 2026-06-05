"""
Model registry — loads from pricing.json, not hardcoded.

Pricing data lives in pricing.json alongside this file.
Run `python -m llm_optimizer.cli update-pricing` to refresh from provider docs.

⚠️  Pricing warning: provider pricing changes without notice.
    The _meta.last_updated field in pricing.json shows when data was last verified.
    Cost estimates may be stale. Always verify at the provider's pricing page
    before making financial decisions based on this library's estimates.
"""

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Optional


class TaskComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    EXPERT = "expert"


class Provider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"


@dataclass
class ModelConfig:
    model_id: str
    provider: Provider
    display_name: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    cached_input_cost_per_1m: float
    context_window: int
    supports_caching: bool = True
    supports_batch: bool = True
    complexity_tiers: list = field(default_factory=list)
    max_output_tokens: int = 8192

    def estimate_cost(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        fresh_input = input_tokens - cached_tokens
        cost = (
            (fresh_input / 1_000_000) * self.input_cost_per_1m
            + (cached_tokens / 1_000_000) * self.cached_input_cost_per_1m
            + (output_tokens / 1_000_000) * self.output_cost_per_1m
        )
        return round(cost, 8)

    def cache_savings(self, input_tokens: int) -> float:
        full_cost = (input_tokens / 1_000_000) * self.input_cost_per_1m
        cached_cost = (input_tokens / 1_000_000) * self.cached_input_cost_per_1m
        return round(full_cost - cached_cost, 8)


_PRICING_PATH = Path(__file__).parent / "pricing.json"
_STALE_DAYS = 30  # Warn if pricing data is older than this


def _load_pricing() -> tuple[dict[str, ModelConfig], dict]:
    """Load models from pricing.json. Warns if data is stale."""
    with open(_PRICING_PATH) as f:
        data = json.load(f)

    meta = data.get("_meta", {})
    last_updated_str = meta.get("last_updated", "unknown")

    # Warn if pricing is stale
    try:
        last_updated = datetime.strptime(last_updated_str, "%Y-%m-%d").date()
        age_days = (date.today() - last_updated).days
        if age_days > _STALE_DAYS:
            warnings.warn(
                f"llm_optimizer pricing data is {age_days} days old (last updated: {last_updated_str}). "
                f"Cost estimates may be inaccurate. Run: python -m llm_optimizer.cli update-pricing",
                UserWarning,
                stacklevel=3,
            )
    except ValueError:
        pass

    tier_map = {t.value: t for t in TaskComplexity}
    provider_map = {p.value: p for p in Provider}

    models: dict[str, ModelConfig] = {}
    for key, m in data.get("models", {}).items():
        models[key] = ModelConfig(
            model_id=m["model_id"],
            provider=provider_map[m["provider"]],
            display_name=m["display_name"],
            input_cost_per_1m=m["input_cost_per_1m"],
            output_cost_per_1m=m["output_cost_per_1m"],
            cached_input_cost_per_1m=m["cached_input_cost_per_1m"],
            context_window=m["context_window"],
            supports_caching=m.get("supports_caching", True),
            supports_batch=m.get("supports_batch", True),
            complexity_tiers=[tier_map[t] for t in m.get("complexity_tiers", [])],
            max_output_tokens=m.get("max_output_tokens", 8192),
        )
    return models, meta


MODELS, PRICING_META = _load_pricing()


def pricing_info() -> dict:
    """Return metadata about the current pricing data."""
    return {
        "version": PRICING_META.get("version"),
        "last_updated": PRICING_META.get("last_updated"),
        "sources": PRICING_META.get("sources", {}),
        "warning": PRICING_META.get("warning"),
    }


# Tier → cheapest capable model per provider
TIER_DEFAULTS: dict[TaskComplexity, dict[Provider, str]] = {
    TaskComplexity.SIMPLE: {
        Provider.ANTHROPIC: "claude-haiku-4-5",
        Provider.OPENAI: "gpt-5-mini",
        Provider.GOOGLE: "gemini-flash",
    },
    TaskComplexity.MEDIUM: {
        Provider.ANTHROPIC: "claude-haiku-4-5",
        Provider.OPENAI: "gpt-5-mini",
        Provider.GOOGLE: "gemini-flash",
    },
    TaskComplexity.COMPLEX: {
        Provider.ANTHROPIC: "claude-sonnet-4",
        Provider.OPENAI: "gpt-5",
        Provider.GOOGLE: "gemini-pro",
    },
    TaskComplexity.EXPERT: {
        Provider.ANTHROPIC: "claude-opus-4",
        Provider.OPENAI: "gpt-5",
        Provider.GOOGLE: "gemini-pro",
    },
}
