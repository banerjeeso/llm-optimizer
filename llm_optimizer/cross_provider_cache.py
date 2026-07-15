"""
Cross-Provider Cache — shares cached responses across providers.

If a prompt was run on Anthropic direct and the user switches to Bedrock,
the cached response is returned without hitting either API.

Cache key is based on prompt content only — not the provider or model ID.
This means:
  - Anthropic → Bedrock: cache hit ✅
  - Bedrock → OpenAI: cache hit ✅
  - Haiku → Opus (same prompt): cache hit ✅ (be careful with quality-sensitive tasks)

For quality-sensitive tasks, set provider_agnostic=False to scope cache
to the same provider, or model_agnostic=False to scope to the same model.

Usage:
    from llm_optimizer import OptimizedClient, Provider

    client = OptimizedClient(
        anthropic_client=...,
        enable_cross_provider_cache=True,
    )

    # First call on Anthropic
    r1 = client.complete(messages=[...], preferred_provider=Provider.ANTHROPIC)

    # Switches to Bedrock — still gets cache hit
    r2 = client.complete(messages=[...], preferred_provider=Provider.BEDROCK)
"""

import hashlib
import json
import time
import threading
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CrossProviderEntry:
    prompt_hash: str
    response_text: str
    source_provider: str
    source_model: str
    input_tokens: int
    output_tokens: int
    original_cost: float
    cached_at: float
    hit_count: int = 0


class CrossProviderCache:
    """
    Provider-agnostic response cache.

    Shares cached responses across Anthropic, Bedrock, OpenAI, and Google.
    Cache key derived from prompt content only — not provider or model.

    Args:
        provider_agnostic:  Share across all providers (default: True)
        model_agnostic:     Share across all models (default: True)
                            Set False to scope to same model family
        ttl_seconds:        Cache entry lifetime
        max_entries:        Maximum entries in cache
    """

    def __init__(
        self,
        provider_agnostic: bool = True,
        model_agnostic: bool = True,
        ttl_seconds: int = 3600,
        max_entries: int = 1000,
    ):
        self.provider_agnostic = provider_agnostic
        self.model_agnostic = model_agnostic
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._cache: dict[str, CrossProviderEntry] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._cost_saved = 0.0
        self._provider_hits: dict[str, int] = {}  # tracks cross-provider hits

    def make_key(
        self,
        messages: list[dict],
        system: Optional[Any] = None,
        model_key: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> str:
        """Generate cache key — provider/model scoping controlled by flags."""
        payload: dict = {"messages": messages}
        if system:
            payload["system"] = str(system)
        if not self.model_agnostic and model_key:
            payload["model"] = model_key
        if not self.provider_agnostic and provider:
            payload["provider"] = provider

        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def get(self, key: str, requesting_provider: Optional[str] = None) -> Optional[CrossProviderEntry]:
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                self._misses += 1
                return None
            if time.time() - entry.cached_at > self.ttl:
                del self._cache[key]
                self._misses += 1
                return None

            entry.hit_count += 1
            self._hits += 1
            self._cost_saved += entry.original_cost

            # Track cross-provider hits
            if requesting_provider and requesting_provider != entry.source_provider:
                key_label = f"{entry.source_provider}→{requesting_provider}"
                self._provider_hits[key_label] = self._provider_hits.get(key_label, 0) + 1

            return entry

    def set(
        self,
        key: str,
        response_text: str,
        source_provider: str,
        source_model: str,
        input_tokens: int,
        output_tokens: int,
        original_cost: float,
    ):
        with self._lock:
            # Evict oldest if over limit
            if len(self._cache) >= self.max_entries:
                oldest_key = min(self._cache, key=lambda k: self._cache[k].cached_at)
                del self._cache[oldest_key]

            self._cache[key] = CrossProviderEntry(
                prompt_hash=key,
                response_text=response_text,
                source_provider=source_provider,
                source_model=source_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                original_cost=original_cost,
                cached_at=time.time(),
            )

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(self._hits / total * 100, 1) if total > 0 else 0.0,
            "cost_saved_usd": round(self._cost_saved, 6),
            "entries": len(self._cache),
            "cross_provider_hits": self._provider_hits,
            "provider_agnostic": self.provider_agnostic,
            "model_agnostic": self.model_agnostic,
        }

    def clear(self):
        with self._lock:
            self._cache.clear()
