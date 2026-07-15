"""
AWS Bedrock Prompt Caching — translates Anthropic cache_control blocks
to Bedrock's CachePoint format automatically.

Key differences between Anthropic and Bedrock caching:

  Anthropic API:
    - Format:   {"cache_control": {"type": "ephemeral", "ttl": "1h"}}
    - Automatic caching: supported (single top-level breakpoint)
    - Min tokens: 1,024 for Claude 4.5 models
    - Max breakpoints: 4

  AWS Bedrock:
    - Format:   {"cachePoint": {"type": "default", "ttl": "1h"}}
    - Automatic caching: NOT supported (must use explicit checkpoints)
    - Min tokens: 4,096 for Claude 4.5 models (higher than direct API)
    - Max breakpoints: 4
    - Max cached tokens: 32,000

  What this module does:
    1. Detects cache_control blocks in Anthropic-format messages
    2. Translates them to Bedrock cachePoint blocks
    3. Injects cachePoint blocks at optimal positions when none exist
    4. Respects Bedrock's 4-checkpoint and 4,096 token minimums
    5. Preserves TTL settings (5m or 1h) — LiteLLM has a bug that drops this

  Result: Bedrock users get the same automatic caching behavior as direct
  Anthropic API users, with no code changes required.

Usage:
    from llm_optimizer.bedrock_cache import BedrockCacheManager

    cache_mgr = BedrockCacheManager(enabled=True, ttl="5m")

    # Translate existing cache_control blocks to cachePoint
    messages, system = cache_mgr.translate(messages, system)

    # Or inject optimal cachePoints from scratch
    messages, system = cache_mgr.inject(messages, system, estimated_tokens={})
"""

from typing import Any, Optional
from .tracker import estimate_tokens

# Bedrock caching constraints
BEDROCK_MAX_CHECKPOINTS = 4
BEDROCK_MIN_TOKENS_CLAUDE_45 = 4096   # Claude Haiku/Sonnet/Opus 4.5
BEDROCK_MIN_TOKENS_OLDER = 1024       # Claude 3.x models
BEDROCK_MAX_CACHED_TOKENS = 32_000

VALID_TTLS = ("5m", "1h")
MODELS_WITH_1H_TTL = {
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
}


def make_cache_point(ttl: str = "5m") -> dict:
    """Create a Bedrock cachePoint block."""
    if ttl not in VALID_TTLS:
        ttl = "5m"
    return {"cachePoint": {"type": "default", "ttl": ttl}}


def strip_cache_control(messages: list[dict]) -> list[dict]:
    """
    Remove Anthropic cache_control blocks from messages.
    Used when Bedrock doesn't support caching for a given model.
    """
    clean = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            clean_blocks = [
                {k: v for k, v in block.items() if k != "cache_control"}
                if isinstance(block, dict) else block
                for block in content
            ]
            clean.append({**msg, "content": clean_blocks})
        else:
            clean.append(msg)
    return clean


def translate_cache_control_to_cachepoint(
    content: list[dict],
    ttl: str = "5m",
    checkpoints_used: list[int] = None,
) -> tuple[list[dict], int]:
    """
    Translate Anthropic cache_control blocks to Bedrock cachePoint blocks.

    Strategy:
      - For each block with cache_control, replace it with the block
        (minus cache_control) followed by a cachePoint block
      - Respect the 4-checkpoint limit
      - Preserve TTL from the original cache_control if present

    Returns:
        (translated_content, checkpoints_added)
    """
    if checkpoints_used is None:
        checkpoints_used = [0]

    result = []
    checkpoints_added = 0

    for block in content:
        if not isinstance(block, dict):
            result.append(block)
            continue

        cache_ctrl = block.get("cache_control")
        clean_block = {k: v for k, v in block.items() if k != "cache_control"}
        result.append(clean_block)

        if cache_ctrl and checkpoints_used[0] < BEDROCK_MAX_CHECKPOINTS:
            # Preserve TTL from original cache_control if valid
            original_ttl = cache_ctrl.get("ttl", ttl)
            if original_ttl not in VALID_TTLS:
                original_ttl = ttl
            result.append(make_cache_point(original_ttl))
            checkpoints_used[0] += 1
            checkpoints_added += 1

    return result, checkpoints_added


class BedrockCacheManager:
    """
    Manages prompt caching for AWS Bedrock requests.

    Handles the translation from Anthropic's cache_control format
    to Bedrock's cachePoint format, and optionally injects optimal
    cache checkpoints when none are present.

    Args:
        enabled:      Enable caching (default: True)
        ttl:          Cache TTL — "5m" or "1h" (default: "5m")
        auto_inject:  Inject cachePoints even when cache_control not present
        min_tokens:   Min tokens before injecting a checkpoint
    """

    def __init__(
        self,
        enabled: bool = True,
        ttl: str = "5m",
        auto_inject: bool = True,
        min_tokens: int = BEDROCK_MIN_TOKENS_CLAUDE_45,
    ):
        self.enabled = enabled
        self.ttl = ttl if ttl in VALID_TTLS else "5m"
        self.auto_inject = auto_inject
        self.min_tokens = min_tokens
        self._checkpoints_used = 0

    def prepare(
        self,
        messages: list[dict],
        system: Optional[Any] = None,
        model_key: Optional[str] = None,
    ) -> tuple[list[dict], Optional[list[dict]]]:
        """
        Prepare messages and system prompt for Bedrock with caching.

        Steps:
          1. Convert system str → list of blocks
          2. Translate any cache_control → cachePoint
          3. If auto_inject and no checkpoints yet, inject at optimal positions
          4. Respect 4-checkpoint limit

        Returns:
            (messages_with_cachepoints, system_with_cachepoints)
        """
        if not self.enabled:
            return strip_cache_control(messages), self._normalize_system(system)

        # Determine TTL based on model
        effective_ttl = self.ttl
        if model_key and model_key not in MODELS_WITH_1H_TTL and self.ttl == "1h":
            effective_ttl = "5m"  # fall back for models without 1h support

        checkpoints_counter = [0]

        # Process system prompt
        system_blocks = self._normalize_system(system)
        if system_blocks:
            translated_system, _ = translate_cache_control_to_cachepoint(
                system_blocks, ttl=effective_ttl, checkpoints_used=checkpoints_counter
            )
            system_blocks = translated_system

            # Auto-inject: if system is large enough and no checkpoint yet
            if (self.auto_inject
                    and checkpoints_counter[0] == 0
                    and self._estimate_block_tokens(system_blocks) >= self.min_tokens):
                system_blocks = self._inject_at_end(
                    system_blocks, effective_ttl, checkpoints_counter
                )

        # Process messages
        processed_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                translated, _ = translate_cache_control_to_cachepoint(
                    content, ttl=effective_ttl, checkpoints_used=checkpoints_counter
                )
                processed_messages.append({**msg, "content": translated})
            else:
                processed_messages.append(msg)

        # Auto-inject into long conversation history
        if (self.auto_inject
                and checkpoints_counter[0] < BEDROCK_MAX_CHECKPOINTS
                and len(processed_messages) > 6):
            processed_messages = self._inject_history_checkpoint(
                processed_messages, effective_ttl, checkpoints_counter
            )

        self._checkpoints_used = checkpoints_counter[0]
        return processed_messages, system_blocks

    def checkpoints_used(self) -> int:
        """Return number of cache checkpoints used in last prepare() call."""
        return self._checkpoints_used

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _normalize_system(system: Optional[Any]) -> Optional[list[dict]]:
        """Convert system prompt to list of content blocks."""
        if system is None:
            return None
        if isinstance(system, str):
            return [{"type": "text", "text": system}]
        if isinstance(system, list):
            return list(system)
        return None

    @staticmethod
    def _inject_at_end(
        blocks: list[dict],
        ttl: str,
        counter: list[int],
    ) -> list[dict]:
        """Append a cachePoint block at the end of the block list."""
        if counter[0] >= BEDROCK_MAX_CHECKPOINTS:
            return blocks
        result = list(blocks)
        result.append(make_cache_point(ttl))
        counter[0] += 1
        return result

    def _inject_history_checkpoint(
        self,
        messages: list[dict],
        ttl: str,
        counter: list[int],
    ) -> list[dict]:
        """
        Inject a cachePoint at ~2/3 through conversation history.
        Stable prefix = everything up to 2/3 = good cache candidate.
        """
        if counter[0] >= BEDROCK_MAX_CHECKPOINTS:
            return messages

        inject_idx = int(len(messages) * 0.66)
        msg = messages[inject_idx]
        content = msg.get("content", "")

        if isinstance(content, str) and estimate_tokens(content) >= self.min_tokens:
            new_content = [
                {"type": "text", "text": content},
                make_cache_point(ttl),
            ]
            messages[inject_idx] = {**msg, "content": new_content}
            counter[0] += 1
        elif isinstance(content, list):
            content_copy = list(content)
            content_copy.append(make_cache_point(ttl))
            messages[inject_idx] = {**msg, "content": content_copy}
            counter[0] += 1

        return messages

    @staticmethod
    def _estimate_block_tokens(blocks: list[dict]) -> int:
        """Estimate total tokens in a list of content blocks."""
        total = 0
        for block in blocks:
            if isinstance(block, dict):
                total += estimate_tokens(block.get("text", ""))
        return total
