"""
Model registry with pricing data and capability tiers.
Prices are per 1M tokens (USD). Updated 2026.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskComplexity(Enum):
    SIMPLE = "simple"        # Classification, extraction, yes/no
    MEDIUM = "medium"        # Summarization, translation, QA
    COMPLEX = "complex"      # Reasoning, code generation, analysis
    EXPERT = "expert"        # Multi-step reasoning, research, advanced code


class Provider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"


@dataclass
class ModelConfig:
    model_id: str
    provider: Provider
    display_name: str
    input_cost_per_1m: float        # USD per 1M input tokens
    output_cost_per_1m: float       # USD per 1M output tokens
    cached_input_cost_per_1m: float # USD per 1M cached input tokens
    context_window: int             # Max tokens
    supports_caching: bool = True
    supports_batch: bool = True
    complexity_tiers: list = field(default_factory=list)  # Which tasks this model handles well
    max_output_tokens: int = 8192

    def estimate_cost(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        """Calculate cost for a given request."""
        fresh_input = input_tokens - cached_tokens
        cost = (
            (fresh_input / 1_000_000) * self.input_cost_per_1m
            + (cached_tokens / 1_000_000) * self.cached_input_cost_per_1m
            + (output_tokens / 1_000_000) * self.output_cost_per_1m
        )
        return round(cost, 8)

    def cache_savings(self, input_tokens: int) -> float:
        """How much you save by caching all input tokens."""
        full_cost = (input_tokens / 1_000_000) * self.input_cost_per_1m
        cached_cost = (input_tokens / 1_000_000) * self.cached_input_cost_per_1m
        return round(full_cost - cached_cost, 8)


# ─── Model Registry ────────────────────────────────────────────────────────────

MODELS: dict[str, ModelConfig] = {
    # Anthropic
    "claude-haiku-4-5": ModelConfig(
        model_id="claude-haiku-4-5-20251001",
        provider=Provider.ANTHROPIC,
        display_name="Claude Haiku 4.5",
        input_cost_per_1m=1.0,
        output_cost_per_1m=5.0,
        cached_input_cost_per_1m=0.1,
        context_window=200_000,
        complexity_tiers=[TaskComplexity.SIMPLE, TaskComplexity.MEDIUM],
    ),
    "claude-haiku-3-5": ModelConfig(
        model_id="claude-haiku-3-5-20241022",
        provider=Provider.ANTHROPIC,
        display_name="Claude Haiku 3.5",
        input_cost_per_1m=0.8,
        output_cost_per_1m=4.0,
        cached_input_cost_per_1m=0.08,
        context_window=200_000,
        complexity_tiers=[TaskComplexity.SIMPLE, TaskComplexity.MEDIUM],
    ),
    "claude-sonnet-4": ModelConfig(
        model_id="claude-sonnet-4-5",
        provider=Provider.ANTHROPIC,
        display_name="Claude Sonnet 4.5",
        input_cost_per_1m=3.0,
        output_cost_per_1m=15.0,
        cached_input_cost_per_1m=0.3,
        context_window=200_000,
        complexity_tiers=[TaskComplexity.SIMPLE, TaskComplexity.MEDIUM, TaskComplexity.COMPLEX],
    ),
    "claude-opus-4": ModelConfig(
        model_id="claude-opus-4-5",
        provider=Provider.ANTHROPIC,
        display_name="Claude Opus 4.5",
        input_cost_per_1m=5.0,
        output_cost_per_1m=25.0,
        cached_input_cost_per_1m=0.5,
        context_window=200_000,
        complexity_tiers=[TaskComplexity.SIMPLE, TaskComplexity.MEDIUM, TaskComplexity.COMPLEX, TaskComplexity.EXPERT],
    ),
    # OpenAI
    "gpt-5-mini": ModelConfig(
        model_id="gpt-5-mini",
        provider=Provider.OPENAI,
        display_name="GPT-5 Mini",
        input_cost_per_1m=0.25,
        output_cost_per_1m=2.0,
        cached_input_cost_per_1m=0.025,
        context_window=400_000,
        complexity_tiers=[TaskComplexity.SIMPLE, TaskComplexity.MEDIUM],
    ),
    "gpt-5": ModelConfig(
        model_id="gpt-5",
        provider=Provider.OPENAI,
        display_name="GPT-5",
        input_cost_per_1m=2.5,
        output_cost_per_1m=15.0,
        cached_input_cost_per_1m=0.25,
        context_window=1_000_000,
        complexity_tiers=[TaskComplexity.SIMPLE, TaskComplexity.MEDIUM, TaskComplexity.COMPLEX, TaskComplexity.EXPERT],
    ),
    # Google
    "gemini-flash": ModelConfig(
        model_id="gemini-2.5-flash",
        provider=Provider.GOOGLE,
        display_name="Gemini 2.5 Flash",
        input_cost_per_1m=0.3,
        output_cost_per_1m=2.5,
        cached_input_cost_per_1m=0.03,
        context_window=1_000_000,
        complexity_tiers=[TaskComplexity.SIMPLE, TaskComplexity.MEDIUM],
    ),
    "gemini-pro": ModelConfig(
        model_id="gemini-2.5-pro",
        provider=Provider.GOOGLE,
        display_name="Gemini 2.5 Pro",
        input_cost_per_1m=1.25,
        output_cost_per_1m=10.0,
        cached_input_cost_per_1m=0.125,
        context_window=2_000_000,
        complexity_tiers=[TaskComplexity.SIMPLE, TaskComplexity.MEDIUM, TaskComplexity.COMPLEX, TaskComplexity.EXPERT],
    ),
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
