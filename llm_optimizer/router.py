"""
Smart Model Router — automatically selects the cheapest model
capable of handling a task's complexity.
"""

import re
from typing import Optional

from .models import (
    ModelConfig, Provider, TaskComplexity,
    MODELS, TIER_DEFAULTS
)
from .tracker import estimate_tokens


# Keywords that signal task complexity
COMPLEXITY_SIGNALS = {
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
        "write code", "debug", "implement", "function", "class", "algorithm",
        "analyze", "compare", "evaluate", "critique", "design", "architecture",
        "generate a report", "detailed explanation", "step by step",
    ],
    TaskComplexity.EXPERT: [
        "research", "reasoning", "multi-step", "complex algorithm", "proof",
        "mathematical", "scientific", "advanced", "optimize code", "refactor",
        "system design", "security audit", "comprehensive analysis",
    ],
}

# Token thresholds for bumping complexity up
LONG_PROMPT_COMPLEXITY_BUMP = {
    5000: TaskComplexity.MEDIUM,    # > 5K tokens → at least medium
    15000: TaskComplexity.COMPLEX,  # > 15K tokens → at least complex
}


class ModelRouter:
    """
    Routes requests to the optimal (cheapest capable) model.

    Can be used in two modes:
      - auto: analyzes prompt complexity and routes automatically
      - manual: you specify complexity tier, it finds cheapest model
    """

    def __init__(
        self,
        preferred_provider: Provider = Provider.ANTHROPIC,
        allow_fallback: bool = True,
        cost_first: bool = True,          # prefer cheapest model
        override_map: Optional[dict] = None,  # complexity → model_key overrides
    ):
        self.preferred_provider = preferred_provider
        self.allow_fallback = allow_fallback
        self.cost_first = cost_first
        self.override_map = override_map or {}

    def route(
        self,
        prompt: str,
        complexity: Optional[TaskComplexity] = None,
        provider: Optional[Provider] = None,
        force_model: Optional[str] = None,
    ) -> tuple[str, ModelConfig, TaskComplexity]:
        """
        Determine the best model for a request.

        Args:
            prompt: The prompt text (used for complexity detection)
            complexity: Override auto-detection with explicit tier
            provider: Override preferred provider
            force_model: Always use this model key regardless of routing

        Returns:
            (model_key, model_config, detected_complexity)
        """
        if force_model:
            model = MODELS.get(force_model)
            if not model:
                raise ValueError(f"Unknown model key: {force_model}. Available: {list(MODELS.keys())}")
            comp = complexity or TaskComplexity.COMPLEX
            return force_model, model, comp

        # Detect complexity
        detected = complexity or self.detect_complexity(prompt)

        # Apply override map
        if detected in self.override_map:
            model_key = self.override_map[detected]
            return model_key, MODELS[model_key], detected

        # Find model from tier defaults
        use_provider = provider or self.preferred_provider
        model_key = self._select_model(detected, use_provider)
        model = MODELS[model_key]

        return model_key, model, detected

    def detect_complexity(self, prompt: str) -> TaskComplexity:
        """
        Heuristically detect task complexity from prompt text.

        Rules (in priority order):
          1. Token count thresholds (very long prompts are inherently complex)
          2. Expert-level keywords
          3. Complex-level keywords
          4. Medium-level keywords
          5. Default: SIMPLE
        """
        lower = prompt.lower()
        token_count = estimate_tokens(prompt)

        # Check token-count thresholds first
        detected_by_length = None
        for threshold, min_complexity in sorted(LONG_PROMPT_COMPLEXITY_BUMP.items(), reverse=True):
            if token_count > threshold:
                detected_by_length = min_complexity
                break

        # Check keyword signals (most specific first) — use word boundaries to avoid
        # false matches like "classify" matching "class"
        detected_by_keyword = None
        for tier in [TaskComplexity.EXPERT, TaskComplexity.COMPLEX, TaskComplexity.MEDIUM, TaskComplexity.SIMPLE]:
            keywords = COMPLEXITY_SIGNALS.get(tier, [])
            if any(re.search(r'\b' + re.escape(kw) + r'\b', lower) for kw in keywords):
                detected_by_keyword = tier
                break

        # Take the higher of the two signals
        if detected_by_length and detected_by_keyword:
            tier_order = [TaskComplexity.SIMPLE, TaskComplexity.MEDIUM, TaskComplexity.COMPLEX, TaskComplexity.EXPERT]
            return max(detected_by_length, detected_by_keyword, key=lambda t: tier_order.index(t))
        return detected_by_length or detected_by_keyword or TaskComplexity.SIMPLE

    def _select_model(self, complexity: TaskComplexity, provider: Provider) -> str:
        """Select model key for a given complexity and provider."""
        tier_map = TIER_DEFAULTS.get(complexity, {})
        model_key = tier_map.get(provider)

        if model_key and model_key in MODELS:
            return model_key

        # Fallback to any available model for this provider
        if self.allow_fallback:
            for key, model in MODELS.items():
                if model.provider == provider and complexity in model.complexity_tiers:
                    return key
            # Last resort: any model that handles this complexity
            for key, model in MODELS.items():
                if complexity in model.complexity_tiers:
                    return key

        raise RuntimeError(f"No model found for complexity={complexity}, provider={provider}")

    def estimate_savings(self, prompt: str, expensive_model_key: str) -> dict:
        """
        Show how much you could save by routing to the optimal model.
        """
        model_key, model, complexity = self.route(prompt)
        expensive = MODELS.get(expensive_model_key)
        if not expensive:
            return {}

        tokens = estimate_tokens(prompt)
        est_output = int(tokens * 0.3)  # rough output estimate

        optimal_cost = model.estimate_cost(tokens, est_output)
        expensive_cost = expensive.estimate_cost(tokens, est_output)
        savings = expensive_cost - optimal_cost

        return {
            "detected_complexity": complexity.value,
            "recommended_model": model_key,
            "recommended_model_cost": optimal_cost,
            "expensive_model": expensive_model_key,
            "expensive_model_cost": expensive_cost,
            "savings_usd": round(savings, 8),
            "savings_pct": round(savings / expensive_cost * 100, 1) if expensive_cost > 0 else 0,
        }
