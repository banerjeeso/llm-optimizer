"""
Semantic Cache — returns cached responses for semantically similar prompts.

Unlike the output cache (exact hash match), this cache uses TF-IDF cosine
similarity to find "close enough" prompts and return their cached responses.

Saves 100% of API cost on slight phrasing variations:
  "What is the capital of France?" ≈ "Capital of France?" ≈ "France's capital?"

No external dependencies — uses pure Python TF-IDF + cosine similarity.
For higher accuracy at scale, swap the backend for FastEmbed or FAISS.

Usage:
    from llm_optimizer import OptimizedClient

    client = OptimizedClient(
        anthropic_client=...,
        enable_semantic_cache=True,
        semantic_cache_threshold=0.92,  # 0-1, higher = stricter matching
        semantic_cache_ttl=3600,
    )

    # First call — hits API
    r1 = client.complete(messages=[{"role":"user","content":"Capital of France?"}])

    # Second call — semantically similar, served from cache at zero cost
    r2 = client.complete(messages=[{"role":"user","content":"What is France's capital?"}])
    
    print(client.semantic_cache_stats())
    # {"hits": 1, "misses": 1, "hit_rate_pct": 50.0, "cost_saved_usd": 0.000012}
"""

import math
import re
import time
import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SemanticCacheEntry:
    prompt_text: str
    response_text: str
    model: str
    input_tokens: int
    output_tokens: int
    original_cost: float
    cached_at: float
    tfidf_vector: dict  # term → tfidf score


# Common English stopwords — excluded from TF-IDF
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "can", "this", "that", "it", "what", "which", "who", "how", "i", "you",
    "he", "she", "we", "they", "not", "no", "so", "if", "as", "than",
}


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer — lowercase words, no stopwords."""
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def _build_tfidf(text: str, corpus_df: dict[str, int], corpus_size: int) -> dict[str, float]:
    """
    Build a TF-IDF vector for a text string.
    corpus_df: document frequency of each term in the cache
    corpus_size: number of documents in cache
    """
    tokens = _tokenize(text)
    if not tokens:
        return {}

    # TF: term frequency
    tf: dict[str, float] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    for t in tf:
        tf[t] /= len(tokens)

    # TF-IDF: weight by inverse document frequency
    vector: dict[str, float] = {}
    for t, freq in tf.items():
        df = corpus_df.get(t, 0)
        idf = math.log((corpus_size + 1) / (df + 1)) + 1
        vector[t] = freq * idf

    # L2 normalize
    norm = math.sqrt(sum(v ** 2 for v in vector.values()))
    if norm > 0:
        vector = {t: v / norm for t, v in vector.items()}

    return vector


def _cosine_similarity(v1: dict[str, float], v2: dict[str, float]) -> float:
    """Cosine similarity between two TF-IDF vectors."""
    if not v1 or not v2:
        return 0.0
    shared = set(v1.keys()) & set(v2.keys())
    dot = sum(v1[t] * v2[t] for t in shared)
    return min(1.0, dot)  # cap at 1.0 due to float precision


def _extract_prompt_text(messages: list[dict], system: Optional[str | list] = None) -> str:
    """Extract all text from messages and system for semantic comparison."""
    parts = []
    if system:
        if isinstance(system, str):
            parts.append(system)
        elif isinstance(system, list):
            parts.extend(b.get("text", "") for b in system if isinstance(b, dict))

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
    return " ".join(parts)


class SemanticCache:
    """
    Fuzzy prompt cache using TF-IDF cosine similarity.

    Returns cached responses when a new prompt is semantically similar
    to a previously cached prompt, based on a configurable threshold.

    Default threshold of 0.92 catches paraphrases and minor wording changes
    while avoiding false positives on meaningfully different questions.

    Args:
        threshold:    Minimum cosine similarity for a cache hit (0.0–1.0)
                      0.92 recommended. Lower = more hits, more false positives.
        ttl_seconds:  Cache entry lifetime in seconds
        max_entries:  Maximum number of entries in cache
    """

    def __init__(
        self,
        threshold: float = 0.92,
        ttl_seconds: int = 3600,
        max_entries: int = 500,
    ):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold
        self.ttl = ttl_seconds
        self.max_entries = max_entries

        self._entries: list[SemanticCacheEntry] = []
        self._corpus_df: dict[str, int] = {}  # document frequency
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._cost_saved = 0.0

    def get(
        self,
        messages: list[dict],
        system: Optional[str | list] = None,
    ) -> Optional[SemanticCacheEntry]:
        """
        Find a semantically similar cached response.
        Returns None on miss. On hit, returns the cached entry.
        """
        prompt_text = _extract_prompt_text(messages, system)
        with self._lock:
            self._evict_expired()
            if not self._entries:
                self._misses += 1
                return None

            query_vec = _build_tfidf(prompt_text, self._corpus_df, len(self._entries))
            best_score = 0.0
            best_entry = None

            for entry in self._entries:
                score = _cosine_similarity(query_vec, entry.tfidf_vector)
                if score > best_score:
                    best_score = score
                    best_entry = entry

            if best_score >= self.threshold and best_entry:
                self._hits += 1
                self._cost_saved += best_entry.original_cost
                return best_entry

            self._misses += 1
            return None

    def set(
        self,
        messages: list[dict],
        response_text: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        original_cost: float,
        system: Optional[str | list] = None,
    ):
        """Store a response in the semantic cache."""
        prompt_text = _extract_prompt_text(messages, system)
        with self._lock:
            self._evict_expired()

            # Update corpus document frequencies
            tokens = set(_tokenize(prompt_text))
            for t in tokens:
                self._corpus_df[t] = self._corpus_df.get(t, 0) + 1

            # Rebuild all TF-IDF vectors with updated corpus
            # (small cache makes this affordable)
            for entry in self._entries:
                entry.tfidf_vector = _build_tfidf(
                    entry.prompt_text, self._corpus_df, len(self._entries) + 1
                )

            tfidf_vec = _build_tfidf(prompt_text, self._corpus_df, len(self._entries) + 1)

            entry = SemanticCacheEntry(
                prompt_text=prompt_text,
                response_text=response_text,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                original_cost=original_cost,
                cached_at=time.time(),
                tfidf_vector=tfidf_vec,
            )

            self._entries.append(entry)

            # Evict oldest if over max size
            if len(self._entries) > self.max_entries:
                self._entries.pop(0)

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(self._hits / total * 100, 1) if total > 0 else 0.0,
            "cost_saved_usd": round(self._cost_saved, 6),
            "entries": len(self._entries),
            "threshold": self.threshold,
        }

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._corpus_df.clear()

    def _evict_expired(self):
        now = time.time()
        active = [e for e in self._entries if now - e.cached_at <= self.ttl]
        if len(active) < len(self._entries):
            # Rebuild corpus DF from active entries only
            self._corpus_df = {}
            for entry in active:
                tokens = set(_tokenize(entry.prompt_text))
                for t in tokens:
                    self._corpus_df[t] = self._corpus_df.get(t, 0) + 1
        self._entries = active
