"""
Prompt Caching — automatically inject cache_control breakpoints
for Anthropic Claude and manage cache-friendly message structures.

Claude supports up to 4 cache breakpoints. Content must be >= 1024 tokens
for a cache breakpoint to be effective (shorter content is ignored by API).

For OpenAI (automatic caching on prefix ≥ 1024 tokens), no extra markup needed.
"""

import hashlib
from typing import Any, Optional

from .tracker import estimate_tokens

# Minimum tokens for a cache block to be worth caching
ANTHROPIC_CACHE_MIN_TOKENS = 1024
MAX_CACHE_BREAKPOINTS = 4  # Anthropic limit


class CacheManager:
    """
    Manages prompt caching strategies.

    For Anthropic: injects `cache_control` blocks at optimal positions.
    For OpenAI: ensures system prompt is placed first (auto-cached).
    For Google: manages context caching via API (when context ≥ 32K tokens).
    """

    def __init__(self, enabled: bool = True, min_tokens_to_cache: int = ANTHROPIC_CACHE_MIN_TOKENS):
        self.enabled = enabled
        self.min_tokens_to_cache = min_tokens_to_cache
        self._cache_hit_registry: dict[str, int] = {}  # hash → hit count

    def prepare_anthropic_messages(
        self,
        messages: list[dict],
        system: Optional[str | list] = None,
        large_documents: Optional[list[str]] = None,
    ) -> tuple[list[dict], Optional[list[dict]]]:
        """
        Inject cache_control into Anthropic messages for maximum cache reuse.

        Strategy:
          1. Cache the system prompt if ≥ min_tokens
          2. Cache large document blocks in user messages
          3. Cache the last few assistant turns in long conversations
          4. Respect the 4-breakpoint limit

        Returns:
            (messages_with_cache, system_with_cache)
        """
        if not self.enabled:
            return messages, self._wrap_system(system)

        breakpoints_used = 0
        processed_messages = [m.copy() for m in messages]

        # 1. Cache system prompt
        system_blocks = self._wrap_system(system)
        if system_blocks and breakpoints_used < MAX_CACHE_BREAKPOINTS:
            system_text = " ".join(b.get("text", "") for b in system_blocks if b.get("type") == "text")
            if estimate_tokens(system_text) >= self.min_tokens_to_cache:
                system_blocks[-1]["cache_control"] = {"type": "ephemeral"}
                breakpoints_used += 1

        # 2. Cache large documents injected into messages
        if large_documents:
            for i, msg in enumerate(processed_messages):
                if breakpoints_used >= MAX_CACHE_BREAKPOINTS:
                    break
                content = msg.get("content", "")
                if isinstance(content, str) and estimate_tokens(content) >= self.min_tokens_to_cache:
                    processed_messages[i]["content"] = [
                        {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                    ]
                    breakpoints_used += 1

        # 3. For long conversations, cache older turns to avoid re-processing
        if len(processed_messages) > 6 and breakpoints_used < MAX_CACHE_BREAKPOINTS:
            # Cache at ~2/3 of conversation history (stable prefix)
            cache_index = int(len(processed_messages) * 0.66)
            msg = processed_messages[cache_index]
            content = msg.get("content", "")
            if isinstance(content, str) and estimate_tokens(content) >= self.min_tokens_to_cache:
                processed_messages[cache_index]["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ]
                breakpoints_used += 1
            elif isinstance(content, list) and content:
                last_block = content[-1]
                if last_block.get("type") == "text":
                    block_text = last_block.get("text", "")
                    if estimate_tokens(block_text) >= self.min_tokens_to_cache:
                        content[-1] = {**last_block, "cache_control": {"type": "ephemeral"}}
                        processed_messages[cache_index]["content"] = content
                        breakpoints_used += 1

        return processed_messages, system_blocks

    def wrap_document_for_caching(self, document_text: str, label: str = "") -> dict:
        """
        Wrap a large document as a cacheable Anthropic content block.
        Use this when injecting reference documents into messages.
        """
        block: dict[str, Any] = {"type": "text", "text": document_text}
        if estimate_tokens(document_text) >= self.min_tokens_to_cache:
            block["cache_control"] = {"type": "ephemeral"}
        return block

    def get_cache_key(self, content: str) -> str:
        """Generate a stable hash key for content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def track_hit(self, cache_key: str):
        """Track how many times a cached prompt is reused."""
        self._cache_hit_registry[cache_key] = self._cache_hit_registry.get(cache_key, 0) + 1

    def hit_count(self, cache_key: str) -> int:
        return self._cache_hit_registry.get(cache_key, 0)

    @staticmethod
    def _wrap_system(system: Optional[str | list]) -> Optional[list[dict]]:
        if system is None:
            return None
        if isinstance(system, str):
            return [{"type": "text", "text": system}]
        return list(system)


def build_cached_system_prompt(
    base_instructions: str,
    few_shot_examples: Optional[str] = None,
    knowledge_base: Optional[str] = None,
) -> list[dict]:
    """
    Build an optimally structured system prompt with cache breakpoints.

    Structure (most stable → least stable):
      [cache1] Base instructions (rarely changes)
      [cache2] Few-shot examples (changes infrequently)
      [cache3] Knowledge base / documents (changes per session)

    This ensures maximum cache reuse across requests.
    """
    blocks = []
    breakpoints = 0

    if base_instructions:
        block: dict = {"type": "text", "text": base_instructions}
        if estimate_tokens(base_instructions) >= ANTHROPIC_CACHE_MIN_TOKENS and breakpoints < MAX_CACHE_BREAKPOINTS:
            block["cache_control"] = {"type": "ephemeral"}
            breakpoints += 1
        blocks.append(block)

    if few_shot_examples:
        block = {"type": "text", "text": few_shot_examples}
        if estimate_tokens(few_shot_examples) >= ANTHROPIC_CACHE_MIN_TOKENS and breakpoints < MAX_CACHE_BREAKPOINTS:
            block["cache_control"] = {"type": "ephemeral"}
            breakpoints += 1
        blocks.append(block)

    if knowledge_base:
        block = {"type": "text", "text": knowledge_base}
        if estimate_tokens(knowledge_base) >= ANTHROPIC_CACHE_MIN_TOKENS and breakpoints < MAX_CACHE_BREAKPOINTS:
            block["cache_control"] = {"type": "ephemeral"}
            breakpoints += 1
        blocks.append(block)

    return blocks
