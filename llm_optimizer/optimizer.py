"""
Prompt Optimizer — reduces token count while preserving semantic meaning.

Techniques:
  - Remove redundant whitespace and formatting
  - Strip filler phrases and hedge words
  - Compress verbose instructions
  - Abbreviate common patterns (JSON structures, code boilerplate)
  - Trim conversation history (keep most recent + summarize old)
"""

import re
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


# Filler phrases that add no semantic value
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
    r'\bin order to\b',             # → "to"
    r'\bdue to the fact that\b',    # → "because"
    r'\bfor the purpose of\b',      # → "for"
    r'\bat this point in time\b',   # → "now"
    r'\bin the event that\b',       # → "if"
]

# Verbose → concise replacements
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
    Reduces token count in prompts and system messages.
    Non-destructive: always returns the optimized text, but
    you can inspect OptimizationResult to see what changed.
    """

    def __init__(
        self,
        enabled: bool = True,
        aggressive: bool = False,       # Apply more aggressive compression
        preserve_code: bool = True,     # Don't modify code blocks
        max_history_turns: int = 10,    # Max conversation turns to keep verbatim
    ):
        self.enabled = enabled
        self.aggressive = aggressive
        self.preserve_code = preserve_code
        self.max_history_turns = max_history_turns

    def optimize(self, text: str) -> OptimizationResult:
        """Optimize a single text string."""
        if not self.enabled or not text:
            tokens = estimate_tokens(text)
            return OptimizationResult(
                original_text=text,
                optimized_text=text,
                original_tokens=tokens,
                optimized_tokens=tokens,
                tokens_saved=0,
                savings_pct=0.0,
                techniques_applied=[],
            )

        original_tokens = estimate_tokens(text)
        optimized = text
        techniques: list[str] = []

        # Extract and protect code blocks
        code_blocks: dict[str, str] = {}
        if self.preserve_code:
            optimized, code_blocks = self._extract_code_blocks(optimized)

        # Apply optimizations
        new_text, new_techniques = self._apply_text_optimizations(optimized)
        if new_text != optimized:
            optimized = new_text
            techniques.extend(new_techniques)

        # Restore code blocks
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
        """
        Optimize a list of chat messages.
        Also applies history truncation if conversation is long.

        Returns:
            (optimized_messages, total_tokens_saved)
        """
        if not self.enabled:
            return messages, 0

        total_saved = 0
        optimized = []

        # Trim history if needed
        messages, was_trimmed = self._trim_history(messages)

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

        # 1. Normalize whitespace
        new_text = re.sub(r'[ \t]+', ' ', text)              # Multiple spaces → one
        new_text = re.sub(r'\n{3,}', '\n\n', new_text)       # 3+ newlines → 2
        new_text = re.sub(r'[ \t]+\n', '\n', new_text)       # Trailing spaces
        new_text = new_text.strip()
        if new_text != text:
            techniques.append("whitespace_normalization")

        # 2. Remove filler phrases
        before = new_text
        for pattern in FILLER_PHRASES:
            new_text = re.sub(pattern, '', new_text, flags=re.IGNORECASE)
        if new_text != before:
            techniques.append("filler_phrase_removal")

        # 3. Verbose → concise replacements
        before = new_text
        for pattern, replacement in VERBOSE_REPLACEMENTS:
            new_text = re.sub(pattern, replacement, new_text, flags=re.IGNORECASE)
        if new_text != before:
            techniques.append("verbose_phrase_compression")

        if self.aggressive:
            # 4. Remove meta-commentary (e.g. "The following is a...")
            before = new_text
            new_text = re.sub(
                r'^(The following (is|are|provides)|Here (is|are|follows)|Below (is|are))[\s\w]+:',
                '',
                new_text,
                flags=re.MULTILINE | re.IGNORECASE,
            )
            if new_text != before:
                techniques.append("meta_commentary_removal")

            # 5. Compress repetitive list markers (- - - → -)
            new_text = re.sub(r'(\s*[-•]\s*){2,}', '\n- ', new_text)

        # Final whitespace cleanup
        new_text = re.sub(r'\n{3,}', '\n\n', new_text).strip()

        return new_text, techniques

    def _extract_code_blocks(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace code blocks with placeholders to avoid modifying them."""
        code_blocks = {}
        counter = [0]

        def replace(m):
            key = f"__CODE_BLOCK_{counter[0]}__"
            code_blocks[key] = m.group(0)
            counter[0] += 1
            return key

        # Match fenced code blocks (```...```) and inline code (`...`)
        result = re.sub(r'```[\s\S]*?```', replace, text)
        result = re.sub(r'`[^`]+`', replace, result)
        return result, code_blocks

    def _trim_history(self, messages: list[dict]) -> tuple[list[dict], bool]:
        """
        Trim conversation history to max_history_turns.
        Keeps: system messages + last N turns.
        """
        if len(messages) <= self.max_history_turns * 2:
            return messages, False

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        # Keep the most recent turns
        keep_count = self.max_history_turns * 2
        trimmed = non_system[-keep_count:]

        return system_msgs + trimmed, True


class DocumentCompressor:
    """
    Compresses long documents before sending to LLM.
    Reduces cost by extracting only the relevant portions.
    """

    def __init__(self, max_tokens: int = 4000, strategy: str = "smart"):
        """
        Args:
            max_tokens: Target max tokens for compressed output
            strategy: "truncate" | "smart" | "chunk"
        """
        self.max_tokens = max_tokens
        self.strategy = strategy

    def compress(self, document: str, query: Optional[str] = None) -> tuple[str, int]:
        """
        Compress a document to fit within token budget.

        Returns:
            (compressed_text, tokens_saved)
        """
        original_tokens = estimate_tokens(document)
        if original_tokens <= self.max_tokens:
            return document, 0

        if self.strategy == "truncate":
            compressed = self._truncate(document)
        elif self.strategy == "smart":
            compressed = self._smart_extract(document, query)
        else:  # chunk
            compressed = self._chunk_first(document)

        compressed_tokens = estimate_tokens(compressed)
        tokens_saved = original_tokens - compressed_tokens
        return compressed, tokens_saved

    def _truncate(self, text: str) -> str:
        """Simple truncation to max_tokens."""
        words = text.split()
        # ~1.3 tokens per word
        max_words = int(self.max_tokens / 1.3)
        if len(words) > max_words:
            truncated = " ".join(words[:max_words])
            return truncated + "\n\n[... document truncated ...]"
        return text

    def _smart_extract(self, text: str, query: Optional[str] = None) -> str:
        """
        Extract most relevant paragraphs.
        Without a query: keep beginning + end (executive summary style).
        With a query: score paragraphs by keyword overlap.
        """
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

        if not paragraphs:
            return self._truncate(text)

        max_words = int(self.max_tokens / 1.3)

        if query:
            query_words = set(re.findall(r'\b\w+\b', query.lower()))
            scored = []
            for i, para in enumerate(paragraphs):
                para_words = set(re.findall(r'\b\w+\b', para.lower()))
                overlap = len(query_words & para_words)
                # Favor paragraphs near start
                position_bonus = 1.0 if i < 3 else (0.5 if i < 10 else 0.2)
                score = overlap * position_bonus
                scored.append((score, i, para))
            scored.sort(key=lambda x: (-x[0], x[1]))  # sort by score, preserve order
        else:
            # Without query: keep first 60% + last 20%
            n = len(paragraphs)
            keep_start = int(n * 0.6)
            keep_end = int(n * 0.2)
            selected_indices = set(range(keep_start)) | set(range(n - keep_end, n))
            scored = [(0, i, p) for i, p in enumerate(paragraphs) if i in selected_indices]

        # Build result respecting token budget
        result_paragraphs = []
        word_count = 0
        for _, idx, para in sorted(scored, key=lambda x: x[1]):
            para_words = len(para.split())
            if word_count + para_words > max_words:
                break
            result_paragraphs.append((idx, para))
            word_count += para_words

        result_paragraphs.sort(key=lambda x: x[0])
        return "\n\n".join(p for _, p in result_paragraphs)

    def _chunk_first(self, text: str) -> str:
        """Return only the first chunk up to max_tokens."""
        return self._truncate(text)
