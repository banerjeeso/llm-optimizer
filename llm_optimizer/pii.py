"""
PII Masking Middleware — detects and masks sensitive data before sending to LLM APIs.

Replaces PII with reversible tokens before the request and restores
original values in the response. Enterprise accounts typically require
this before sending data to public APIs.

Detected patterns:
  - Email addresses      → [EMAIL_1]
  - Phone numbers        → [PHONE_1]
  - Credit card numbers  → [CARD_1]
  - SSN                  → [SSN_1]
  - IP addresses         → [IP_1]
  - Names (capitalized)  → [NAME_1] (heuristic — may have false positives)
  - Dates                → [DATE_1]
  - URLs                 → [URL_1]
  - Custom patterns      → [CUSTOM_1]

Usage:
    from llm_optimizer import OptimizedClient

    client = OptimizedClient(
        anthropic_client=...,
        enable_pii_masking=True,
        pii_restore_in_response=True,   # replace tokens back in response
    )

    # PII is masked before sending, restored in response
    response = client.complete(
        messages=[{"role": "user", "content":
            "John Smith (john@example.com) owes $500 on card 4111-1111-1111-1111"
        }],
        system="Summarize the debt information.",
    )
    # Sent to API: "[NAME_1] ([EMAIL_1]) owes $500 on card [CARD_1]"
    # Response restored: mentions John Smith, john@example.com, etc.

    # Audit what was masked
    print(client.pii_stats())
"""

import re
import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MaskingSession:
    """Tracks one request's masking tokens for restoration."""
    token_map: dict[str, str] = field(default_factory=dict)  # token → original
    reverse_map: dict[str, str] = field(default_factory=dict)  # original → token
    counters: dict[str, int] = field(default_factory=dict)

    def add(self, category: str, original: str) -> str:
        """Create or reuse a token for a PII value."""
        if original in self.reverse_map:
            return self.reverse_map[original]
        count = self.counters.get(category, 0) + 1
        self.counters[category] = count
        token = f"[{category}_{count}]"
        self.token_map[token] = original
        self.reverse_map[original] = token
        return token

    def restore(self, text: str) -> str:
        """Replace tokens with original values."""
        for token, original in self.token_map.items():
            text = text.replace(token, original)
        return text

    def masked_count(self) -> int:
        return len(self.token_map)


# ─── PII Detection Patterns ────────────────────────────────────────────────────

PII_PATTERNS = [
    # Email addresses
    ("EMAIL", re.compile(
        r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'
    )),
    # Credit card numbers (major formats)
    ("CARD", re.compile(
        r'\b(?:4[0-9]{12}(?:[0-9]{3})?|'         # Visa
        r'5[1-5][0-9]{14}|'                        # MasterCard
        r'3[47][0-9]{13}|'                         # Amex
        r'6(?:011|5[0-9]{2})[0-9]{12}|'           # Discover
        r'(?:\d{4}[-\s]){3}\d{4})\b'              # Formatted with dashes/spaces
    )),
    # SSN (US)
    ("SSN", re.compile(
        r'\b\d{3}[-\s]\d{2}[-\s]\d{4}\b'
    )),
    # Phone numbers (US + international)
    ("PHONE", re.compile(
        r'\b(?:\+?1[-.\s]?)?'
        r'(?:\(?\d{3}\)?[-.\s]?)'
        r'\d{3}[-.\s]?\d{4}\b'
    )),
    # IP addresses
    ("IP", re.compile(
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    )),
    # URLs
    ("URL", re.compile(
        r'https?://[^\s<>"\'{}|\\^`\[\]]+'
    )),
    # Dates (common formats)
    ("DATE", re.compile(
        r'\b(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|'
        r'\d{4}[/\-]\d{1,2}[/\-]\d{1,2}|'
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{1,2},?\s+\d{4})\b',
        re.IGNORECASE
    )),
]

# Name detection (heuristic — Title Case words not in common English vocab)
_COMMON_WORDS = {
    "The", "A", "An", "In", "On", "At", "To", "For", "Of", "And", "Or",
    "But", "Is", "Are", "Was", "Were", "Be", "Been", "Have", "Has", "Had",
    "Do", "Does", "Did", "Will", "Would", "Could", "Should", "May", "Can",
    "This", "That", "These", "Those", "It", "Its", "From", "With", "By",
    "As", "If", "Not", "No", "So", "Than", "Then", "When", "Where", "Which",
    "Who", "What", "How", "All", "Any", "Both", "Each", "Few", "More",
    "Most", "Other", "Some", "Such", "New", "Old", "First", "Last", "Long",
    "Little", "Own", "Right", "Big", "High", "Small", "Large", "Next", "Early",
    "Young", "Important", "Public", "Private", "Real", "Best", "Free", "Sure",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "June", "July", "August",
    "September", "October", "November", "December",
    "American", "English", "French", "German", "Spanish", "Chinese",
    "Google", "Apple", "Microsoft", "Amazon", "Facebook", "Twitter",
}

NAME_PATTERN = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')


class PIIMasker:
    """
    Masks PII in text before sending to LLM APIs.

    Creates a per-request session that tracks masked values for restoration.
    Thread-safe — each request gets its own MaskingSession.

    Args:
        mask_emails:       Mask email addresses (default: True)
        mask_phones:       Mask phone numbers (default: True)
        mask_cards:        Mask credit card numbers (default: True)
        mask_ssn:          Mask SSNs (default: True)
        mask_ips:          Mask IP addresses (default: False)
        mask_urls:         Mask URLs (default: False)
        mask_dates:        Mask dates (default: False)
        mask_names:        Mask person names — heuristic (default: False)
        custom_patterns:   Dict of {category: compiled_regex} for custom PII
    """

    def __init__(
        self,
        mask_emails: bool = True,
        mask_phones: bool = True,
        mask_cards: bool = True,
        mask_ssn: bool = True,
        mask_ips: bool = False,
        mask_urls: bool = False,
        mask_dates: bool = False,
        mask_names: bool = False,
        custom_patterns: Optional[dict] = None,
    ):
        self._active_patterns: list[tuple[str, re.Pattern]] = []

        flags = {
            "EMAIL": mask_emails,
            "CARD": mask_cards,
            "SSN": mask_ssn,
            "PHONE": mask_phones,
            "IP": mask_ips,
            "URL": mask_urls,
            "DATE": mask_dates,
        }

        for category, pattern in PII_PATTERNS:
            if flags.get(category, True):
                self._active_patterns.append((category, pattern))

        self._mask_names = mask_names
        if custom_patterns:
            for cat, pat in custom_patterns.items():
                self._active_patterns.append((cat.upper(), pat))

        self._lock = threading.Lock()
        self._total_masked = 0
        self._requests_with_pii = 0

    def mask_messages(
        self,
        messages: list[dict],
        system: Optional[str | list] = None,
    ) -> tuple[list[dict], Optional[str | list], MaskingSession]:
        """
        Mask PII in messages and system prompt.

        Returns:
            (masked_messages, masked_system, session)
            Keep the session to restore PII in the response.
        """
        session = MaskingSession()

        # Mask system
        masked_system = None
        if system:
            if isinstance(system, str):
                masked_system = self._mask_text(system, session)
            elif isinstance(system, list):
                masked_system = [
                    {**b, "text": self._mask_text(b["text"], session)}
                    if isinstance(b, dict) and b.get("type") == "text"
                    else b
                    for b in system
                ]

        # Mask messages
        masked_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                masked_messages.append({**msg, "content": self._mask_text(content, session)})
            elif isinstance(content, list):
                masked_blocks = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        masked_blocks.append({**block, "text": self._mask_text(block["text"], session)})
                    else:
                        masked_blocks.append(block)
                masked_messages.append({**msg, "content": masked_blocks})
            else:
                masked_messages.append(msg)

        with self._lock:
            count = session.masked_count()
            self._total_masked += count
            if count > 0:
                self._requests_with_pii += 1

        return masked_messages, masked_system, session

    def restore_response(self, response_text: str, session: MaskingSession) -> str:
        """Restore original PII values in the model's response."""
        return session.restore(response_text)

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_pii_values_masked": self._total_masked,
                "requests_with_pii": self._requests_with_pii,
                "active_pattern_categories": [cat for cat, _ in self._active_patterns],
                "name_detection": self._mask_names,
            }

    def _mask_text(self, text: str, session: MaskingSession) -> str:
        """Apply all active masking patterns to text."""
        # Apply regex patterns (most specific first — cards before phones)
        for category, pattern in self._active_patterns:
            text = pattern.sub(
                lambda m, cat=category, s=session: s.add(cat, m.group()),
                text,
            )

        # Name detection (heuristic — applied last to avoid masking already-tokenized text)
        if self._mask_names:
            def replace_name(m):
                name = m.group(1)
                # Skip if it's a common word or already a token
                words = name.split()
                if any(w in _COMMON_WORDS for w in words):
                    return m.group()
                if name.startswith("[") and name.endswith("]"):
                    return m.group()
                return session.add("NAME", name)
            text = NAME_PATTERN.sub(replace_name, text)

        return text
