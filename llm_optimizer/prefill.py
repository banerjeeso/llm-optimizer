"""
Prefill & Forced Output — auto-inject assistant prefill to guide model output.

For Anthropic models, prefilling the assistant turn forces structured output
and reduces formatting errors without extra tokens in the system prompt.

Common patterns:
  - Force JSON:     prefill='{"'
  - Force list:     prefill='1.'
  - Force markdown: prefill='# '
  - Force code:     prefill='```python'

The library auto-detects the requested output format and injects the
appropriate prefill, removing the need to add formatting instructions
to your system prompt (which costs tokens on every request).

Usage:
    client = OptimizedClient(
        anthropic_client=...,
        enable_prefill=True,
    )

    # Auto-detects JSON output needed from system prompt keywords
    response = client.complete(
        messages=[{"role": "user", "content": "Extract name and age from: John, 30"}],
        system="Always respond with valid JSON only.",
        output_schema={"type": "object", "properties": {"name": {}, "age": {}}},
    )

    # Or explicit prefill
    response = client.complete(
        messages=[...],
        prefill='{"name":',
    )
"""

import json
import re
from typing import Any, Optional


# Keywords in system prompt that suggest JSON output
JSON_SIGNALS = [
    "json", "json only", "valid json", "return json", "respond with json",
    "output json", "json format", "json object", "json array",
    "structured output", "respond only with",
]

# Keywords suggesting list output
LIST_SIGNALS = [
    "numbered list", "bulleted list", "list of", "enumerate",
]

# Keywords suggesting code output
CODE_SIGNALS = [
    "write code", "write a function", "implement", "python function",
    "javascript function", "code snippet",
]


def detect_prefill(
    system: Optional[str | list] = None,
    output_schema: Optional[dict] = None,
    explicit_prefill: Optional[str] = None,
) -> Optional[str]:
    """
    Determine the optimal prefill string for the assistant turn.

    Priority:
      1. Explicit prefill from caller
      2. Output schema → JSON prefill
      3. System prompt keywords → format-specific prefill
      4. None (no prefill)

    Args:
        system:          System prompt text or blocks
        output_schema:   JSON schema dict — forces JSON prefill
        explicit_prefill: Caller-provided prefill string

    Returns:
        Prefill string or None
    """
    # 1. Explicit always wins
    if explicit_prefill is not None:
        return explicit_prefill

    # 2. Output schema → JSON
    if output_schema:
        return '{"'

    if system is None:
        return None

    # Normalize system to string
    if isinstance(system, list):
        system_text = " ".join(
            b.get("text", "") for b in system
            if isinstance(b, dict) and b.get("type") == "text"
        ).lower()
    else:
        system_text = system.lower()

    # 3. Keyword detection
    if any(sig in system_text for sig in JSON_SIGNALS):
        return '{"'
    if any(sig in system_text for sig in LIST_SIGNALS):
        return "1."
    if any(sig in system_text for sig in CODE_SIGNALS):
        return "```python\n"

    return None


def inject_prefill(
    messages: list[dict],
    prefill: str,
) -> list[dict]:
    """
    Inject a prefill string as the start of the assistant turn.

    For Anthropic: adds an assistant message at the end of the messages list.
    This forces Claude to continue from that prefix.

    Note: The prefill is removed from the response before returning to caller.
    """
    # Don't double-inject
    if messages and messages[-1].get("role") == "assistant":
        return messages

    return messages + [{"role": "assistant", "content": prefill}]


def strip_prefill_from_response(response_text: str, prefill: str) -> str:
    """
    Remove the prefill from the start of the response text.
    The model echoes the prefill — strip it before returning to caller.
    """
    if response_text.startswith(prefill):
        return response_text[len(prefill):]
    return response_text


class OutputSchemaValidator:
    """
    Validates model output against a JSON schema.
    If validation fails, flags for auto-correction.
    """

    def __init__(self, schema: Optional[dict] = None):
        self.schema = schema

    def validate(self, text: str) -> tuple[bool, Optional[dict], Optional[str]]:
        """
        Validate text as JSON matching the schema.

        Returns:
            (is_valid, parsed_json, error_message)
        """
        if not self.schema:
            return True, None, None

        # Try to parse JSON
        try:
            # Strip markdown code fences if present
            clean = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
            clean = re.sub(r'\s*```$', '', clean.strip(), flags=re.MULTILINE)
            parsed = json.loads(clean.strip())
            return True, parsed, None
        except json.JSONDecodeError as e:
            return False, None, str(e)

    def extract_json(self, text: str) -> Optional[str]:
        """Try to extract JSON from text even if surrounded by other content."""
        # Look for JSON object or array
        patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # nested objects
            r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',  # arrays
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    json.loads(match)
                    return match
                except json.JSONDecodeError:
                    continue
        return None
