"""
Smart Model Router — selects the cheapest model capable of handling a task.

Decision logic (auditable, in priority order):
  1. force_model      → always use this exact model, skip everything else
  2. explicit complexity  → user-supplied TaskComplexity, skip auto-detection
  3. LLM classifier   → call Haiku to assess complexity (opt-in, costs ~10 tokens)
  4. Heuristics       → keyword signals + token-count thresholds (fallback)

Heuristic signals documented below so they can be audited and overridden:
  SIMPLE  : classify, extract, translate, yes/no, sentiment, spam detection
  MEDIUM  : summarize, paraphrase, rewrite, Q&A with short context
  COMPLEX : write code, debug, implement, analyze, design, evaluate
  EXPERT  : multi-step reasoning, research, proofs, security audits, system design

Token-count bumps (applied on top of keyword signals):
  > 5,000 tokens  → at least MEDIUM
  > 15,000 tokens → at least COMPLEX

KNOWN LIMITATIONS of keyword heuristics:
  - Short prompts with complex intent are misclassified.
    Example: "Explain the constitutional implications of this clause" → classified
    as SIMPLE by keywords, but requires expert-level reasoning.
  - Use enable_llm_classifier=True to avoid this class of error at ~10 token cost.
  - Keyword matching is case-insensitive with word boundaries (no substring matches).
"""

import re
import json
from typing import Optional

from .models import (
    ModelConfig, Provider, TaskComplexity,
    MODELS, TIER_DEFAULTS
)
from .tracker import estimate_tokens


# ─── Documented heuristic signals ─────────────────────────────────────────────
# Changing these changes routing behavior. Document any modifications.

COMPLEXITY_SIGNALS: dict[TaskComplexity, list[str]] = {
    TaskComplexity.SIMPLE: [
        "classify", "yes or no", "true or false", "label", "categorize",
        "extract", "is this", "does this", "sentiment", "spam", "translate",
        "detect language", "format", "parse", "convert json", "fix typos",
    ],
    TaskComplexity.MEDIUM: [
        "summarize", "summary", "paraphrase", "rewrite", "edit", "improve",
        "answer the question", "explain briefly", "list the", "what are",
    ],
    TaskComplexity.COMPLEX: [
        "write code", "debug", "implement", "algorithm",
        "analyze", "compare", "evaluate", "critique", "design", "architecture",
        "generate a report", "detailed explanation", "step by step",
    ],
    TaskComplexity.EXPERT: [
        "research", "multi-step", "complex algorithm", "proof",
        "mathematical", "scientific", "advanced", "optimize code", "refactor",
        "system design", "security audit", "comprehensive analysis",
        "constitutional", "implications", "legal analysis",
    ],
}

LONG_PROMPT_COMPLEXITY_BUMP: dict[int, TaskComplexity] = {
    5000: TaskComplexity.MEDIUM,
    15000: TaskComplexity.COMPLEX,
}

TIER_ORDER = [
    TaskComplexity.SIMPLE,
    TaskComplexity.MEDIUM,
    TaskComplexity.COMPLEX,
    TaskComplexity.EXPERT,
]

# Classifier system prompt — fixed, auditable
_CLASSIFIER_SYSTEM = """You are a task complexity classifier for LLM routing.
Classify the user's request into exactly one of: SIMPLE, MEDIUM, COMPLEX, EXPERT.

SIMPLE  : Classification, extraction, yes/no, translation, spam detection, sentiment
MEDIUM  : Summarization, paraphrasing, Q&A over short context, light editing
COMPLEX : Code generation, debugging, analysis, evaluation, multi-paragraph writing
EXPERT  : Multi-step reasoning, legal/constitutional analysis, security audits, research,
          mathematical proofs, system design, anything requiring deep domain knowledge

Reply with a JSON object only. No other text.
{"complexity": "SIMPLE"|"MEDIUM"|"COMPLEX"|"EXPERT", "reason": "<one sentence>"}"""


class ModelRouter:
    """
    Routes requests to the optimal (cheapest capable) model.

    Two detection modes:
      - Heuristic (default): fast, free, but misses intent-heavy short prompts
      - LLM classifier: uses Haiku to assess complexity; costs ~10-20 tokens
        but is significantly more accurate. Enable with enable_llm_classifier=True.

    Override map lets you pin specific complexity tiers to specific models
    without exposing TaskComplexity throughout your codebase.
    """

    def __init__(
        self,
        preferred_provider: Provider = Provider.ANTHROPIC,
        allow_fallback: bool = True,
        enable_llm_classifier: bool = False,   # opt-in; costs tokens but is accurate
        anthropic_client=None,                  # required if enable_llm_classifier=True
        override_map: Optional[dict] = None,    # {TaskComplexity: model_key} escape hatches
    ):
        self.preferred_provider = preferred_provider
        self.allow_fallback = allow_fallback
        self.enable_llm_classifier = enable_llm_classifier
        self._client = anthropic_client
        self.override_map = override_map or {}

        if enable_llm_classifier and not anthropic_client:
            raise ValueError(
                "enable_llm_classifier=True requires an anthropic_client. "
                "Pass anthropic_client=anthropic.Anthropic() to ModelRouter."
            )

    def route(
        self,
        prompt: str,
        complexity: Optional[TaskComplexity] = None,
        provider: Optional[Provider] = None,
        force_model: Optional[str] = None,
    ) -> tuple[str, ModelConfig, TaskComplexity]:
        """
        Determine the best model for a request.

        Priority:
          1. force_model  — bypasses all routing logic
          2. complexity   — user-supplied override (escape hatch; see docstring)
          3. LLM classifier (if enabled)
          4. Heuristics

        Returns:
            (model_key, model_config, detected_complexity)
        """
        # 1. Hard override
        if force_model:
            model = MODELS.get(force_model)
            if not model:
                raise ValueError(
                    f"Unknown model key: '{force_model}'. "
                    f"Available: {list(MODELS.keys())}"
                )
            return force_model, model, complexity or TaskComplexity.COMPLEX

        # 2. Explicit complexity (escape hatch — caller takes responsibility)
        if complexity:
            model_key = self._select_model(complexity, provider or self.preferred_provider)
            return model_key, MODELS[model_key], complexity

        # 3. LLM classifier (accurate, costs ~10-20 tokens)
        if self.enable_llm_classifier:
            detected = self._classify_with_llm(prompt)
        else:
            # 4. Heuristics (fast, free, documented above)
            detected = self._heuristic_detect(prompt)

        # Apply override map
        if detected in self.override_map:
            model_key = self.override_map[detected]
            return model_key, MODELS[model_key], detected

        model_key = self._select_model(detected, provider or self.preferred_provider)
        return model_key, MODELS[model_key], detected

    def detect_complexity(self, prompt: str) -> TaskComplexity:
        """Public method to inspect the routing decision without making an API call."""
        return self._heuristic_detect(prompt)

    def explain(self, prompt: str) -> dict:
        """
        Return a full audit trail of the routing decision.
        Useful for debugging misroutes.
        """
        heuristic = self._heuristic_detect(prompt)
        token_count = estimate_tokens(prompt)

        # Which keywords fired?
        lower = prompt.lower()
        fired: dict[str, list[str]] = {}
        for tier, keywords in COMPLEXITY_SIGNALS.items():
            matches = [kw for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', lower)]
            if matches:
                fired[tier.value] = matches

        bump = None
        for threshold, min_tier in sorted(LONG_PROMPT_COMPLEXITY_BUMP.items(), reverse=True):
            if token_count > threshold:
                bump = {"threshold": threshold, "bumped_to": min_tier.value}
                break

        model_key = self._select_model(heuristic, self.preferred_provider)
        return {
            "detected_complexity": heuristic.value,
            "routed_model": model_key,
            "token_count": token_count,
            "keyword_signals_fired": fired,
            "token_count_bump": bump,
            "llm_classifier_used": False,
            "warning": (
                "Heuristics cannot detect intent-heavy short prompts. "
                "Use enable_llm_classifier=True for high-stakes routing."
            ) if not self.enable_llm_classifier else None,
        }

    def estimate_savings(self, prompt: str, expensive_model_key: str) -> dict:
        model_key, model, complexity = self.route(prompt)
        expensive = MODELS.get(expensive_model_key)
        if not expensive:
            return {"error": f"Unknown model: {expensive_model_key}"}

        tokens = estimate_tokens(prompt)
        est_output = int(tokens * 0.3)
        optimal_cost = model.estimate_cost(tokens, est_output)
        expensive_cost = expensive.estimate_cost(tokens, est_output)
        savings = expensive_cost - optimal_cost

        return {
            "detected_complexity": complexity.value,
            "recommended_model": model_key,
            "recommended_cost": optimal_cost,
            "expensive_model": expensive_model_key,
            "expensive_cost": expensive_cost,
            "savings_usd": round(savings, 8),
            "savings_pct": round(savings / expensive_cost * 100, 1) if expensive_cost > 0 else 0,
        }

    # ─── Internal ──────────────────────────────────────────────────────────────

    def _heuristic_detect(self, prompt: str) -> TaskComplexity:
        """
        Keyword + token-count heuristics.

        LIMITATION: misclassifies short prompts with complex intent.
        The LLM classifier does not have this problem.
        """
        lower = prompt.lower()
        token_count = estimate_tokens(prompt)

        detected_by_length: Optional[TaskComplexity] = None
        for threshold, min_tier in sorted(LONG_PROMPT_COMPLEXITY_BUMP.items(), reverse=True):
            if token_count > threshold:
                detected_by_length = min_tier
                break

        detected_by_keyword: Optional[TaskComplexity] = None
        for tier in [TaskComplexity.EXPERT, TaskComplexity.COMPLEX,
                     TaskComplexity.MEDIUM, TaskComplexity.SIMPLE]:
            keywords = COMPLEXITY_SIGNALS.get(tier, [])
            if any(re.search(r'\b' + re.escape(kw) + r'\b', lower) for kw in keywords):
                detected_by_keyword = tier
                break

        if detected_by_length and detected_by_keyword:
            return max(detected_by_length, detected_by_keyword,
                       key=lambda t: TIER_ORDER.index(t))
        return detected_by_length or detected_by_keyword or TaskComplexity.SIMPLE

    def _classify_with_llm(self, prompt: str) -> TaskComplexity:
        """
        Use Haiku to classify task complexity.
        Cost: ~10-20 input tokens + ~20 output tokens ≈ $0.000015 per call.
        Significantly more accurate than heuristics for intent-heavy prompts.
        """
        try:
            response = self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=60,
                system=_CLASSIFIER_SYSTEM,
                messages=[{"role": "user", "content": prompt[:2000]}],  # cap at 2K
            )
            raw = response.content[0].text.strip()
            data = json.loads(raw)
            label = data.get("complexity", "MEDIUM").upper()
            mapping = {
                "SIMPLE": TaskComplexity.SIMPLE,
                "MEDIUM": TaskComplexity.MEDIUM,
                "COMPLEX": TaskComplexity.COMPLEX,
                "EXPERT": TaskComplexity.EXPERT,
            }
            return mapping.get(label, TaskComplexity.MEDIUM)
        except Exception:
            # Classifier failure → fall back to heuristics silently
            return self._heuristic_detect(prompt)

    def _select_model(self, complexity: TaskComplexity, provider: Provider) -> str:
        tier_map = TIER_DEFAULTS.get(complexity, {})
        model_key = tier_map.get(provider)
        if model_key and model_key in MODELS:
            return model_key

        if self.allow_fallback:
            for key, model in MODELS.items():
                if model.provider == provider and complexity in model.complexity_tiers:
                    return key
            for key, model in MODELS.items():
                if complexity in model.complexity_tiers:
                    return key

        raise RuntimeError(
            f"No model found for complexity={complexity.value}, provider={provider.value}"
        )
