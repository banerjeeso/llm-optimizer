"""
Dynamic Few-Shot Selector — picks the most relevant examples for each query.

Instead of hardcoding all N examples into every prompt (expensive),
this module picks the top-K most relevant examples for the current query
using TF-IDF similarity. Only the relevant examples are injected into
the cacheable prompt area, saving tokens on every request.

Savings example:
  100 examples × 50 tokens = 5,000 tokens per request (hardcoded)
  Top 3 selected  × 50 tokens = 150 tokens per request (dynamic)
  Savings: 4,850 tokens = 97% reduction in few-shot overhead

Usage:
    from llm_optimizer.few_shot import FewShotSelector

    selector = FewShotSelector(top_k=3)

    # Register your example bank once
    selector.add("classify email as spam", "SPAM")
    selector.add("is this message promotional?", "SPAM")
    selector.add("translate hello to french", "Bonjour")
    selector.add("what is the capital of Germany?", "Berlin")

    # At request time — get the top 3 most relevant examples
    examples = selector.select("categorize this email")
    # Returns the 2 spam examples (most relevant) + 1 other

    # Format for injection into system prompt
    formatted = selector.format(examples)
    # "Input: classify email as spam\nOutput: SPAM\n\n..."
"""

import re
import math
import threading
from dataclasses import dataclass
from typing import Optional


_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "this", "that", "it",
    "what", "which", "i", "you", "he", "she", "we", "they", "not", "no",
}


@dataclass
class FewShotExample:
    input_text: str
    output_text: str
    label: Optional[str] = None     # optional category tag for filtering
    tokens: int = 0                  # estimated token cost
    tfidf_vector: dict = None

    def __post_init__(self):
        if self.tfidf_vector is None:
            self.tfidf_vector = {}


def _tokenize(text: str) -> list[str]:
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def _cosine_sim(v1: dict, v2: dict) -> float:
    if not v1 or not v2:
        return 0.0
    shared = set(v1.keys()) & set(v2.keys())
    dot = sum(v1[t] * v2[t] for t in shared)
    return min(1.0, dot)


class FewShotSelector:
    """
    Selects the most relevant few-shot examples for each query.

    Uses TF-IDF cosine similarity to rank examples by relevance.
    No external dependencies required.

    Args:
        top_k:         Number of examples to select per query (default: 3)
        max_tokens:    Token budget for all selected examples combined
        label_filter:  Only select examples with this label (optional)
    """

    def __init__(
        self,
        top_k: int = 3,
        max_tokens: int = 1000,
    ):
        self.top_k = top_k
        self.max_tokens = max_tokens
        self._examples: list[FewShotExample] = []
        self._corpus_df: dict[str, int] = {}
        self._lock = threading.Lock()

    def add(
        self,
        input_text: str,
        output_text: str,
        label: Optional[str] = None,
    ) -> "FewShotSelector":
        """
        Add an example to the bank.
        Returns self for chaining.
        """
        from .tracker import estimate_tokens
        tokens = estimate_tokens(f"{input_text}\n{output_text}")

        example = FewShotExample(
            input_text=input_text,
            output_text=output_text,
            label=label,
            tokens=tokens,
        )

        with self._lock:
            # Update corpus document frequencies
            input_tokens = set(_tokenize(input_text))
            for t in input_tokens:
                self._corpus_df[t] = self._corpus_df.get(t, 0) + 1

            self._examples.append(example)

            # Rebuild all TF-IDF vectors with updated corpus
            n = len(self._examples)
            for ex in self._examples:
                ex.tfidf_vector = self._build_vector(ex.input_text, n)

        return self

    def add_bulk(self, examples: list[tuple]) -> "FewShotSelector":
        """
        Add multiple examples at once.
        examples: list of (input, output) or (input, output, label) tuples
        """
        for item in examples:
            if len(item) == 2:
                self.add(item[0], item[1])
            elif len(item) == 3:
                self.add(item[0], item[1], item[2])
        return self

    def select(
        self,
        query: str,
        label_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[FewShotExample]:
        """
        Select the most relevant examples for a query.

        Args:
            query:        The current user query
            label_filter: Only consider examples with this label
            top_k:        Override instance top_k for this call

        Returns:
            List of top-K most relevant FewShotExample objects
        """
        k = top_k or self.top_k

        with self._lock:
            candidates = [
                ex for ex in self._examples
                if label_filter is None or ex.label == label_filter
            ]

            if not candidates:
                return []

            # Build query vector
            query_vec = self._build_vector(query, len(self._examples))

            # Score each example
            scored = [
                (_cosine_sim(query_vec, ex.tfidf_vector), ex)
                for ex in candidates
            ]
            scored.sort(key=lambda x: -x[0])

            # Fill token budget
            selected = []
            token_budget = self.max_tokens
            for _, ex in scored[:k * 2]:  # consider 2x top_k, filter by budget
                if len(selected) >= k:
                    break
                if ex.tokens <= token_budget:
                    selected.append(ex)
                    token_budget -= ex.tokens

        return selected

    def format(
        self,
        examples: list[FewShotExample],
        input_label: str = "Input",
        output_label: str = "Output",
        separator: str = "\n\n",
    ) -> str:
        """
        Format selected examples as a string for injection into prompt.

        Args:
            examples:     Selected examples from select()
            input_label:  Label for the input field
            output_label: Label for the output field
            separator:    Separator between examples
        """
        if not examples:
            return ""

        parts = []
        for ex in examples:
            parts.append(f"{input_label}: {ex.input_text}\n{output_label}: {ex.output_text}")

        return separator.join(parts)

    def stats(self) -> dict:
        return {
            "total_examples": len(self._examples),
            "top_k": self.top_k,
            "max_tokens_per_selection": self.max_tokens,
            "labels": list({ex.label for ex in self._examples if ex.label}),
        }

    def _build_vector(self, text: str, corpus_size: int) -> dict[str, float]:
        tokens = _tokenize(text)
        if not tokens:
            return {}
        tf: dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1.0 / len(tokens)
        vector = {}
        for t, freq in tf.items():
            idf = math.log((corpus_size + 1) / (self._corpus_df.get(t, 0) + 1)) + 1
            vector[t] = freq * idf
        norm = math.sqrt(sum(v ** 2 for v in vector.values()))
        if norm > 0:
            vector = {t: v / norm for t, v in vector.items()}
        return vector
