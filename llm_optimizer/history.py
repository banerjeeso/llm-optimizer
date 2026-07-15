"""
History Summarizer — compresses long conversation history using a cheap model.

Instead of sending 20 turns verbatim (expensive), summarizes turns 1-15
with Haiku, then sends [summary + last 5 turns] (cheap).

Savings: 60-80% on token cost for long conversations.

Usage:
    client = OptimizedClient(
        anthropic_client=...,
        enable_history_summarization=True,
        history_summarization_threshold=10,  # summarize when > 10 turns
        history_keep_recent=4,               # always keep last 4 turns verbatim
    )

    # First 10 turns sent verbatim
    # Turn 11+ triggers summarization of older turns
    # Result: [system] + [summary block] + [last 4 turns]
"""

import json
from typing import Optional


SUMMARIZER_SYSTEM = """You are a conversation summarizer. 
Summarize the provided conversation history concisely, preserving:
- Key decisions made
- Important facts established
- Context needed to continue the conversation
- Any unresolved questions

Be concise. Output plain text only."""

SUMMARY_BLOCK_PREFIX = "[CONVERSATION SUMMARY — earlier turns compressed]\n"


class HistorySummarizer:
    """
    Summarizes old conversation turns to reduce token cost.

    How it works:
      1. Monitor conversation length
      2. When turns exceed threshold, take the oldest turns
      3. Call Haiku to summarize them (~$0.001 per summary)
      4. Replace old turns with a single summary message
      5. Append recent turns verbatim

    Cost math example:
      20 turns × 500 tokens = 10,000 input tokens × $3/1M (Sonnet) = $0.03
      Summary + 5 turns = 3,000 tokens = $0.009  →  70% savings per request

    Args:
        client:       anthropic.Anthropic() — used to call Haiku for summarization
        threshold:    Summarize when conversation exceeds this many turns
        keep_recent:  Always keep this many recent turns verbatim
        model:        Model to use for summarization (default: Haiku — cheapest)
    """

    def __init__(
        self,
        client=None,
        threshold: int = 10,
        keep_recent: int = 4,
        model: str = "claude-haiku-4-5-20251001",
    ):
        self._client = client
        self.threshold = threshold
        self.keep_recent = keep_recent
        self.model = model

    def maybe_summarize(self, messages: list[dict]) -> tuple[list[dict], bool]:
        """
        Check if summarization is needed and apply it.

        Args:
            messages: Current conversation history

        Returns:
            (messages, was_summarized)
            was_summarized=True means token count was reduced
        """
        # Filter out system messages — only count user/assistant turns
        system_msgs = [m for m in messages if m.get("role") == "system"]
        convo_msgs = [m for m in messages if m.get("role") != "system"]

        # Count turns (user+assistant pairs)
        turn_count = len(convo_msgs) // 2

        if turn_count <= self.threshold:
            return messages, False

        if not self._client:
            # No client — fall back to simple truncation
            keep = self.keep_recent * 2
            return system_msgs + convo_msgs[-keep:], True

        # Split: summarize old turns, keep recent verbatim
        keep_count = self.keep_recent * 2
        to_summarize = convo_msgs[:-keep_count]
        to_keep = convo_msgs[-keep_count:]

        summary = self._summarize(to_summarize)

        summary_message = {
            "role": "user",
            "content": SUMMARY_BLOCK_PREFIX + summary,
        }

        return system_msgs + [summary_message] + to_keep, True

    def _summarize(self, messages: list[dict]) -> str:
        """Call Haiku to summarize a list of messages."""
        try:
            # Format messages as readable text for summarization
            formatted = []
            for msg in messages:
                role = msg.get("role", "user").upper()
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                formatted.append(f"{role}: {content}")

            conversation_text = "\n\n".join(formatted)

            response = self._client.messages.create(
                model=self.model,
                max_tokens=512,
                system=SUMMARIZER_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": f"Summarize this conversation:\n\n{conversation_text}"
                }],
            )
            return response.content[0].text

        except Exception as e:
            # Summarization failure → fall back to raw text join
            return f"[Summary unavailable: {e}] Earlier conversation had {len(messages)} messages."
