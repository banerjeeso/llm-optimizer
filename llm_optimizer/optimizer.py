"""
Prompt Optimizer and Document Compressor.

IMPORTANT — DocumentCompressor accuracy warning:
  Document compression is a lossy operation. It reduces token count by
  removing content. There is NO guarantee the removed content is irrelevant
  to your query. Use only when:
    (a) you have verified the strategy works for your document type, OR
    (b) you accept that answers may be incomplete or wrong.

  Strategies ranked by accuracy (best → worst):
    "extractive"  — scores paragraphs by TF-IDF overlap with query; best accuracy
    "smart"       — keeps first 60% + last 20% by position; decent for structured docs
    "truncate"    — hard cutoff at max_tokens; worst accuracy, most predictable

  All strategies emit a warning in the returned text so downstream code can
  detect that compression occurred. The caller is responsible for deciding
  whether to surface this to end users.
"""

import re
import math
from dataclasses import dataclass
from typing import Optional

from .tracker import estimate_tokens


@dataclass
class OptimizationResult:
    original_text: str
    optimized_text: str
    original_tokens: int
    optimized_tokens: int
    tokens_saved: int
    savings_pct: float
    techniques_applied: list[str]


# ─── Filler phrases that add no semantic value ─────────────────────────────────

FILLER_PHRASES = [
    r'\bplease note that\b',
    r'\bit is important to note that\b',
    r'\bplease be aware that\b',
    r'\bas you may know\b',
    r'\bas previously mentioned\b',
    r'\bwith that being said\b',
    r'\bwithout further ado\b',
    r'\bI hope this helps\b',
    r'\bfeel free to\b',
    r'\bdue to the fact that\b',
    r'\bfor the purpose of\b',
    r'\bat this point in time\b',
    r'\bin the event that\b',
]

VERBOSE_REPLACEMENTS = [
    (r'\bin order to\b', 'to'),
    (r'\bdue to the fact that\b', 'because'),
    (r'\bfor the purpose of\b', 'for'),
    (r'\bat this point in time\b', 'now'),
    (r'\bin the event that\b', 'if'),
    (r'\bwith regard to\b', 'about'),
    (r'\bwith respect to\b', 'about'),
    (r'\bprior to\b', 'before'),
    (r'\bsubsequent to\b', 'after'),
    (r'\ba large number of\b', 'many'),
    (r'\ba number of\b', 'several'),
    (r'\bthe majority of\b', 'most'),
    (r'\bhas the ability to\b', 'can'),
    (r'\bis able to\b', 'can'),
    (r'\bwas able to\b', 'could'),
    (r'\bwill be able to\b', 'can'),
    (r'\bplease make sure to\b', 'ensure'),
    (r'\bmake sure that you\b', 'ensure you'),
    (r'\bit is necessary to\b', 'you must'),
    (r'\bI would like you to\b', ''),
    (r'\bcould you please\b', ''),
    (r'\bI need you to\b', ''),
    (r'\bCan you please\b', ''),
]


class PromptOptimizer:
    """
    Reduces token count in prompts via text normalization.
    These transforms are low-risk (whitespace, filler phrases, verbose constructs).
    Code blocks are always preserved.
    """

    def __init__(
        self,
        enabled: bool = True,
        aggressive: bool = False,
        preserve_code: bool = True,
        max_history_turns: int = 10,
    ):
        self.enabled = enabled
        self.aggressive = aggressive
        self.preserve_code = preserve_code
        self.max_history_turns = max_history_turns

    def optimize(self, text: str) -> OptimizationResult:
        if not self.enabled or not text:
            tokens = estimate_tokens(text)
            return OptimizationResult(text, text, tokens, tokens, 0, 0.0, [])

        original_tokens = estimate_tokens(text)
        optimized = text
        techniques: list[str] = []

        code_blocks: dict[str, str] = {}
        if self.preserve_code:
            optimized, code_blocks = self._extract_code_blocks(optimized)

        new_text, new_techniques = self._apply_text_optimizations(optimized)
        if new_text != optimized:
            optimized = new_text
            techniques.extend(new_techniques)

        if code_blocks:
            for placeholder, code in code_blocks.items():
                optimized = optimized.replace(placeholder, code)

        optimized_tokens = estimate_tokens(optimized)
        tokens_saved = original_tokens - optimized_tokens
        savings_pct = round(tokens_saved / original_tokens * 100, 1) if original_tokens > 0 else 0

        return OptimizationResult(
            original_text=text,
            optimized_text=optimized,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            tokens_saved=tokens_saved,
            savings_pct=savings_pct,
            techniques_applied=techniques,
        )

    def optimize_messages(self, messages: list[dict]) -> tuple[list[dict], int]:
        if not self.enabled:
            return messages, 0

        total_saved = 0
        optimized = []
        messages, _ = self._trim_history(messages)

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                result = self.optimize(content)
                total_saved += result.tokens_saved
                optimized.append({**msg, "content": result.optimized_text})
            elif isinstance(content, list):
                new_blocks = []
                for block in content:
                    if block.get("type") == "text":
                        result = self.optimize(block["text"])
                        total_saved += result.tokens_saved
                        new_blocks.append({**block, "text": result.optimized_text})
                    else:
                        new_blocks.append(block)
                optimized.append({**msg, "content": new_blocks})
            else:
                optimized.append(msg)

        return optimized, total_saved

    def _apply_text_optimizations(self, text: str) -> tuple[str, list[str]]:
        techniques = []
        new_text = re.sub(r'[ \t]+', ' ', text)
        new_text = re.sub(r'\n{3,}', '\n\n', new_text)
        new_text = re.sub(r'[ \t]+\n', '\n', new_text)
        new_text = new_text.strip()
        if new_text != text:
            techniques.append("whitespace_normalization")

        before = new_text
        for pattern in FILLER_PHRASES:
            new_text = re.sub(pattern, '', new_text, flags=re.IGNORECASE)
        if new_text != before:
            techniques.append("filler_phrase_removal")

        before = new_text
        for pattern, replacement in VERBOSE_REPLACEMENTS:
            new_text = re.sub(pattern, replacement, new_text, flags=re.IGNORECASE)
        if new_text != before:
            techniques.append("verbose_phrase_compression")

        if self.aggressive:
            before = new_text
            new_text = re.sub(
                r'^(The following (is|are|provides)|Here (is|are|follows)|Below (is|are))[\s\w]+:',
                '', new_text, flags=re.MULTILINE | re.IGNORECASE,
            )
            if new_text != before:
                techniques.append("meta_commentary_removal")

        new_text = re.sub(r'\n{3,}', '\n\n', new_text).strip()
        return new_text, techniques

    def _extract_code_blocks(self, text: str) -> tuple[str, dict[str, str]]:
        code_blocks = {}
        counter = [0]

        def replace(m):
            key = f"__CODE_BLOCK_{counter[0]}__"
            code_blocks[key] = m.group(0)
            counter[0] += 1
            return key

        result = re.sub(r'```[\s\S]*?```', replace, text)
        result = re.sub(r'`[^`]+`', replace, result)
        return result, code_blocks

    def _trim_history(self, messages: list[dict]) -> tuple[list[dict], bool]:
        if len(messages) <= self.max_history_turns * 2:
            return messages, False
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]
        trimmed = non_system[-(self.max_history_turns * 2):]
        return system_msgs + trimmed, True


# ─── Document Compressor ───────────────────────────────────────────────────────

COMPRESSION_WARNING = (
    "\n\n[⚠️ COMPRESSION WARNING: This document was truncated to fit within the token "
    "budget. Some content may have been removed. Answers based on this document "
    "may be incomplete. Strategy used: {strategy}]"
)


class DocumentCompressor:
    """
    Compresses documents to fit within a token budget.

    ⚠️  ACCURACY WARNING — read before using:
        Compression removes content. There is no guarantee removed content
        is irrelevant to your query. This feature trades accuracy for cost.

        Recommended use cases:
          - Background/reference documents where you need rough context
          - RAG pipelines where you have re-ranking downstream
          - Cases where you have empirically validated the strategy

        Do NOT use blindly for:
          - Legal or compliance documents (clauses matter)
          - Code files (any removed line may be critical)
          - Any task where completeness is required for correctness

    Strategies:
        "extractive"  Best accuracy. Scores paragraphs by TF-IDF overlap
                      with the query. Requires a query string.
        "smart"       Medium accuracy. Keeps beginning + end of document.
                      Good for structured documents with summary upfront.
        "truncate"    Lowest accuracy. Hard cutoff. Predictable data loss.

    All strategies append a visible warning to the compressed text.
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        strategy: str = "smart",
        warn: bool = True,       # always True; set False only in tests
    ):
        if strategy not in ("extractive", "smart", "truncate"):
            raise ValueError(f"Unknown strategy '{strategy}'. Choose: extractive, smart, truncate")
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.warn = warn

    def compress(self, document: str, query: Optional[str] = None) -> tuple[str, int]:
        """
        Compress document to fit within max_tokens.

        Args:
            document: The source document text
            query:    Query string for relevance-based compression (required for 'extractive')

        Returns:
            (compressed_text, tokens_saved)

        Note:
            Returns original document unchanged if it already fits within max_tokens.
        """
        original_tokens = estimate_tokens(document)
        if original_tokens <= self.max_tokens:
            return document, 0

        if self.strategy == "extractive":
            if not query:
                # Fall back to smart without crashing
                compressed = self._smart_extract(document, query=None)
                effective_strategy = "smart (extractive requested but no query provided)"
            else:
                compressed = self._extractive(document, query)
                effective_strategy = "extractive"
        elif self.strategy == "smart":
            compressed = self._smart_extract(document, query)
            effective_strategy = "smart"
        else:
            compressed = self._truncate(document)
            effective_strategy = "truncate"

        if self.warn:
            compressed += COMPRESSION_WARNING.format(strategy=effective_strategy)

        compressed_tokens = estimate_tokens(compressed)
        tokens_saved = max(0, original_tokens - compressed_tokens)
        return compressed, tokens_saved

    def _truncate(self, text: str) -> str:
        """Hard cutoff. Predictable but loses everything after the limit."""
        words = text.split()
        max_words = int(self.max_tokens / 1.3)
        if len(words) > max_words:
            return " ".join(words[:max_words])
        return text

    def _smart_extract(self, text: str, query: Optional[str]) -> str:
        """
        Position-based extraction: keep beginning (60%) + end (20%) of paragraphs.
        Rationale: most documents front-load key information and end with conclusions.
        Not query-aware — use extractive for better accuracy with a known query.
        """
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        if not paragraphs:
            return self._truncate(text)

        max_words = int(self.max_tokens / 1.3)
        n = len(paragraphs)
        keep_start = int(n * 0.6)
        keep_end = max(1, int(n * 0.2))
        selected = list(range(keep_start)) + list(range(n - keep_end, n))
        selected = sorted(set(selected))

        result, word_count = [], 0
        for i in selected:
            para_words = len(paragraphs[i].split())
            if word_count + para_words > max_words:
                break
            result.append(paragraphs[i])
            word_count += para_words

        return "\n\n".join(result)

    def _extractive(self, text: str, query: str) -> str:
        """
        TF-IDF paragraph scoring against query.
        Most accurate strategy — keeps paragraphs most relevant to the query.

        Algorithm:
          1. Split document into paragraphs
          2. Build TF-IDF scores for each paragraph word
          3. Score each paragraph by overlap with query terms (IDF-weighted)
          4. Sort by score descending, fill token budget, restore original order
        """
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        if not paragraphs:
            return self._truncate(text)

        query_terms = set(re.findall(r'\b\w+\b', query.lower())) - _STOPWORDS

        # Compute IDF across paragraphs
        df: dict[str, int] = {}
        for para in paragraphs:
            terms = set(re.findall(r'\b\w+\b', para.lower()))
            for t in terms:
                df[t] = df.get(t, 0) + 1

        n = len(paragraphs)

        def score(para: str, idx: int) -> float:
            terms = re.findall(r'\b\w+\b', para.lower())
            tf: dict[str, float] = {}
            for t in terms:
                tf[t] = tf.get(t, 0) + 1
            if terms:
                for t in tf:
                    tf[t] /= len(terms)

            s = 0.0
            for qt in query_terms:
                if qt in tf:
                    idf = math.log((n + 1) / (df.get(qt, 0) + 1)) + 1
                    s += tf[qt] * idf
            # Position bonus: first 3 paragraphs are often most important
            position_bonus = 1.5 if idx < 3 else 1.0
            return s * position_bonus

        scored = [(score(p, i), i, p) for i, p in enumerate(paragraphs)]
        scored.sort(key=lambda x: -x[0])

        max_words = int(self.max_tokens / 1.3)
        selected: list[tuple[int, str]] = []
        word_count = 0

        for _, idx, para in scored:
            pw = len(para.split())
            if word_count + pw > max_words:
                break
            selected.append((idx, para))
            word_count += pw

        # Restore original document order
        selected.sort(key=lambda x: x[0])
        return "\n\n".join(p for _, p in selected)


# Common English stopwords — excluded from TF-IDF scoring
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "as", "if", "not", "no", "so", "than",
    "then", "when", "where", "which", "who", "what", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such",
}
