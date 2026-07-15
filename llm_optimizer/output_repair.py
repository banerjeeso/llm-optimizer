"""
Output Auto-Correction — intercepts malformed JSON and fixes it using Haiku.

When a model returns broken JSON despite instructions to return valid JSON,
this module:
  1. Detects the JSON parse error
  2. Passes the broken JSON + error to Haiku (~$0.001 cost)
  3. Gets back valid JSON
  4. Returns it cleanly to caller — no exception raised

Total cost of auto-correction: ~$0.001 per fix vs ~$0.01-0.05 for a full retry.
The fix call uses Haiku regardless of what model generated the original output.

Usage:
    client = OptimizedClient(
        anthropic_client=...,
        enable_output_correction=True,
        output_schema={"type": "object", "properties": {"name": {}, "age": {}}},
    )

    # If model returns broken JSON, it's auto-fixed before reaching caller
    response = client.complete(
        messages=[{"role": "user", "content": "Extract: John is 30 years old"}],
        system="Return JSON with name and age fields.",
    )
    # response.content is guaranteed valid JSON if correction succeeded
"""

import json
import re
from typing import Any, Optional


CORRECTION_SYSTEM = """You are a JSON repair assistant.
The user will give you broken or malformed JSON and the parse error.
Your job is to return ONLY valid, corrected JSON.
Do not explain. Do not wrap in markdown. Return only the JSON."""

CORRECTION_MAX_TOKENS = 512
CORRECTION_MODEL = "claude-haiku-4-5-20251001"


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Try multiple strategies to extract valid JSON from text.

    Strategy order:
      1. Parse text directly
      2. Strip markdown code fences
      3. Find JSON object/array with regex
      4. Find first { and last } and try that substring
    """
    # 1. Direct parse
    try:
        json.loads(text.strip())
        return text.strip()
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences
    clean = re.sub(r'^```(?:json)?\s*\n?', '', text.strip(), flags=re.MULTILINE)
    clean = re.sub(r'\n?```\s*$', '', clean.strip(), flags=re.MULTILINE)
    try:
        json.loads(clean.strip())
        return clean.strip()
    except json.JSONDecodeError:
        pass

    # 3. Find JSON object with regex
    obj_match = re.search(r'\{[\s\S]*\}', text)
    if obj_match:
        try:
            json.loads(obj_match.group())
            return obj_match.group()
        except json.JSONDecodeError:
            pass

    # 4. Find JSON array
    arr_match = re.search(r'\[[\s\S]*\]', text)
    if arr_match:
        try:
            json.loads(arr_match.group())
            return arr_match.group()
        except json.JSONDecodeError:
            pass

    return None


class OutputCorrector:
    """
    Intercepts malformed JSON responses and fixes them using Haiku.

    Args:
        client:       anthropic.Anthropic() client for correction calls
        enabled:      Enable auto-correction (default: True)
        max_attempts: Max correction attempts before giving up (default: 2)
        model:        Model to use for correction (default: Haiku — cheapest)
    """

    def __init__(
        self,
        client=None,
        enabled: bool = True,
        max_attempts: int = 2,
        model: str = CORRECTION_MODEL,
    ):
        self._client = client
        self.enabled = enabled
        self.max_attempts = max_attempts
        self.model = model
        self._corrections_made = 0
        self._corrections_failed = 0

    def correct(
        self,
        response_text: str,
        output_schema: Optional[dict] = None,
        original_error: Optional[str] = None,
    ) -> tuple[str, bool]:
        """
        Attempt to correct a malformed response.

        Args:
            response_text:  The broken response from the model
            output_schema:  Expected JSON schema (optional, improves correction)
            original_error: The parse error message (optional)

        Returns:
            (corrected_text, was_corrected)
        """
        if not self.enabled:
            return response_text, False

        # First try extraction without LLM
        extracted = extract_json_from_text(response_text)
        if extracted:
            return extracted, True

        # Need LLM correction
        if not self._client:
            return response_text, False

        for attempt in range(self.max_attempts):
            try:
                corrected = self._llm_correct(response_text, output_schema, original_error)
                if corrected:
                    # Verify it's valid JSON
                    json.loads(corrected)
                    self._corrections_made += 1
                    return corrected, True
            except (json.JSONDecodeError, Exception):
                continue

        self._corrections_failed += 1
        return response_text, False

    def validate_and_correct(
        self,
        response_text: str,
        output_schema: Optional[dict] = None,
    ) -> tuple[str, bool, Optional[str]]:
        """
        Validate response and correct if needed.

        Returns:
            (final_text, is_valid_json, error_message)
        """
        # Try to parse as-is
        try:
            json.loads(response_text.strip())
            return response_text, True, None
        except json.JSONDecodeError as e:
            error_msg = str(e)

        # Attempt correction
        corrected, was_corrected = self.correct(response_text, output_schema, error_msg)

        try:
            json.loads(corrected.strip())
            return corrected, True, None
        except json.JSONDecodeError as e:
            return corrected, False, str(e)

    def stats(self) -> dict:
        return {
            "corrections_made": self._corrections_made,
            "corrections_failed": self._corrections_failed,
            "success_rate_pct": round(
                self._corrections_made /
                (self._corrections_made + self._corrections_failed) * 100, 1
            ) if (self._corrections_made + self._corrections_failed) > 0 else 0.0,
        }

    def _llm_correct(
        self,
        broken_json: str,
        schema: Optional[dict],
        error: Optional[str],
    ) -> Optional[str]:
        """Call Haiku to fix the broken JSON."""
        prompt_parts = [f"Broken JSON:\n{broken_json}"]
        if error:
            prompt_parts.append(f"Parse error: {error}")
        if schema:
            prompt_parts.append(f"Expected schema: {json.dumps(schema, indent=2)}")
        prompt_parts.append("Return only the corrected JSON:")

        response = self._client.messages.create(
            model=self.model,
            max_tokens=CORRECTION_MAX_TOKENS,
            system=CORRECTION_SYSTEM,
            messages=[{"role": "user", "content": "\n\n".join(prompt_parts)}],
        )

        corrected = response.content[0].text.strip()

        # Strip any markdown the model added
        corrected = re.sub(r'^```(?:json)?\s*\n?', '', corrected, flags=re.MULTILINE)
        corrected = re.sub(r'\n?```\s*$', '', corrected.strip(), flags=re.MULTILINE)
        return corrected.strip()
