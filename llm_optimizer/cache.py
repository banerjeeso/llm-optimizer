"""
Output Cache — returns cached API responses for identical prompts.
Zero API cost on cache hits.

Two backends:
  "memory"  — in-process LRU cache. Fast, zero deps, lost on restart.
  "sqlite"  — persistent SQLite cache. Survives restarts, shareable across processes.

Cache key is a SHA-256 hash of: model_id + system + messages + temperature.
Identical inputs always produce the same key regardless of whitespace or ordering.

Usage:
    from llm_optimizer import OptimizedClient

    # In-memory cache (dev/test)
    client = OptimizedClient(
        anthropic_client=...,
        enable_output_cache=True,
        output_cache_ttl=3600,          # 1 hour
        output_cache_max_size=500,      # max 500 entries
    )

    # SQLite cache (production — survives restarts)
    client = OptimizedClient(
        anthropic_client=...,
        enable_output_cache=True,
        output_cache_backend="sqlite",
        output_cache_path="llm_cache.db",
        output_cache_ttl=86400,         # 24 hours
    )

    # First call — hits API
    r1 = client.complete(messages=[{"role": "user", "content": "What is 2+2?"}])

    # Second call — served from cache, zero cost
    r2 = client.complete(messages=[{"role": "user", "content": "What is 2+2?"}])

    print(client.cache_stats())
    # {"hits": 1, "misses": 1, "hit_rate_pct": 50.0, "cost_saved_usd": 0.000012}
"""

import hashlib
import json
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CachedResponse:
    """Normalized cached response — provider-agnostic."""
    content: str           # The text response
    model: str
    input_tokens: int
    output_tokens: int
    cached_at: float       # Unix timestamp
    original_cost: float   # What the live call would have cost


class OutputCache:
    """
    Caches complete LLM responses by prompt hash.

    Hit rate is highest for:
      - Repeated identical queries (FAQ bots, classification pipelines)
      - Dev/test environments running the same prompts repeatedly
      - Batch jobs with duplicate inputs

    Hit rate is zero for:
      - Conversational apps with unique user inputs
      - Creative generation tasks

    Check cache_stats() to see if it's worth keeping enabled.
    """

    def __init__(
        self,
        backend: str = "memory",           # "memory" | "sqlite"
        ttl_seconds: int = 3600,           # Cache entry lifetime
        max_size: int = 1000,              # Memory backend only — max entries
        db_path: str = "llm_cache.db",    # SQLite backend only
    ):
        if backend not in ("memory", "sqlite"):
            raise ValueError(f"Unknown cache backend: '{backend}'. Choose: memory, sqlite")

        self.backend = backend
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._cost_saved = 0.0

        if backend == "memory":
            self._cache: OrderedDict[str, CachedResponse] = OrderedDict()
        else:
            self._db_path = db_path
            self._init_sqlite()

    def get(self, cache_key: str) -> Optional[CachedResponse]:
        """Return cached response or None on miss."""
        with self._lock:
            result = self._get(cache_key)
            if result:
                self._hits += 1
                self._cost_saved += result.original_cost
            else:
                self._misses += 1
            return result

    def set(self, cache_key: str, response: CachedResponse):
        """Store a response in cache."""
        with self._lock:
            self._set(cache_key, response)

    def stats(self) -> dict:
        """Return cache performance statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(self._hits / total * 100, 1) if total > 0 else 0.0,
            "cost_saved_usd": round(self._cost_saved, 6),
            "backend": self.backend,
            "ttl_seconds": self.ttl,
        }

    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            if self.backend == "memory":
                self._cache.clear()
            else:
                conn = sqlite3.connect(self._db_path)
                conn.execute("DELETE FROM response_cache")
                conn.commit()
                conn.close()

    def invalidate(self, cache_key: str):
        """Remove a specific entry from cache."""
        with self._lock:
            if self.backend == "memory":
                self._cache.pop(cache_key, None)
            else:
                conn = sqlite3.connect(self._db_path)
                conn.execute("DELETE FROM response_cache WHERE key = ?", (cache_key,))
                conn.commit()
                conn.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get(self, key: str) -> Optional[CachedResponse]:
        if self.backend == "memory":
            entry = self._cache.get(key)
            if entry is None:
                return None
            if time.time() - entry.cached_at > self.ttl:
                del self._cache[key]
                return None
            # Move to end (LRU)
            self._cache.move_to_end(key)
            return entry
        else:
            return self._sqlite_get(key)

    def _set(self, key: str, response: CachedResponse):
        if self.backend == "memory":
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = response
            # Evict oldest if over max size
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)
        else:
            self._sqlite_set(key, response)

    def _init_sqlite(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS response_cache (
                key TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cached_at REAL NOT NULL,
                original_cost REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cached_at ON response_cache(cached_at)")
        conn.commit()
        conn.close()

    def _sqlite_get(self, key: str) -> Optional[CachedResponse]:
        try:
            conn = sqlite3.connect(self._db_path)
            row = conn.execute(
                "SELECT content, model, input_tokens, output_tokens, cached_at, original_cost "
                "FROM response_cache WHERE key = ?", (key,)
            ).fetchone()
            conn.close()

            if row is None:
                return None

            content, model, input_t, output_t, cached_at, cost = row

            # Check TTL
            if time.time() - cached_at > self.ttl:
                self.invalidate(key)
                return None

            return CachedResponse(
                content=content, model=model,
                input_tokens=input_t, output_tokens=output_t,
                cached_at=cached_at, original_cost=cost,
            )
        except Exception:
            return None

    def _sqlite_set(self, key: str, response: CachedResponse):
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                INSERT OR REPLACE INTO response_cache
                (key, content, model, input_tokens, output_tokens, cached_at, original_cost)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                key, response.content, response.model,
                response.input_tokens, response.output_tokens,
                response.cached_at, response.original_cost,
            ))
            conn.commit()
            conn.close()
        except Exception:
            pass


def make_cache_key(model_id: str, messages: list, system: Any, temperature: float) -> str:
    """
    Generate a stable cache key from request parameters.
    Identical inputs always produce the same key.
    """
    payload = {
        "model": model_id,
        "messages": messages,
        "system": str(system) if system else None,
        "temperature": temperature,
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def extract_text_from_response(response: Any) -> str:
    """Extract text content from any provider response."""
    # Anthropic / Bedrock
    if hasattr(response, "content"):
        parts = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts)
    # OpenAI
    if hasattr(response, "choices"):
        return response.choices[0].message.content or ""
    return str(response)
