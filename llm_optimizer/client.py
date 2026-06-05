"""
OptimizedClient — main entry point for llm_optimizer.

Streaming support:
    Supported via client.stream(). See docstring below.
    client.complete() does NOT stream. Use client.stream() for user-facing apps.

Error handling and fallback behavior:
    - Rate limit (429):  retries up to max_retries times with exponential backoff
    - Model unavailable: falls back to next capable model if allow_fallback=True
    - Batch API error:   re-queues requests; raises after max_retries
    - Cache miss:        transparent (Anthropic handles this server-side)
    - Classifier fail:   falls back to heuristic routing silently

All failure modes are documented in the LIMITATIONS section of the README.
"""

import time
import warnings
from typing import Any, Optional

from .models import ModelConfig, Provider, TaskComplexity, MODELS, TIER_DEFAULTS
from .caching import CacheManager, build_cached_system_prompt
from .router import ModelRouter
from .optimizer import PromptOptimizer, DocumentCompressor
from .batcher import BatchProcessor
from .tracker import CostTracker, estimate_tokens
from .streaming import StreamResult, build_anthropic_stream, build_openai_stream


class OptimizedClient:
    """
    Drop-in wrapper that applies six cost optimizations transparently.

    Streaming:
        Use client.stream() for user-facing applications. client.complete()
        blocks until the full response is available.

        with client.stream(messages=[...], system="...") as stream:
            for chunk in stream:
                print(chunk, end="", flush=True)

    Error handling:
        By default, rate limit errors are retried with exponential backoff
        (max_retries=3, starting at 1s). Model-level errors fall back to the
        next capable model. Set max_retries=0 to disable retries.

    Fallback behavior:
        - Rate limited      → retry with backoff (up to max_retries)
        - Model overloaded  → fall back to next capable model
        - Batch unavailable → re-queue; raises RuntimeError after max_retries
        - Classifier error  → silently fall back to heuristic routing
    """

    def __init__(
        self,
        anthropic_client=None,
        openai_client=None,
        google_client=None,
        # Feature flags
        enable_caching: bool = True,
        enable_routing: bool = True,
        enable_optimization: bool = True,
        enable_compression: bool = False,   # Opt-in; see DocumentCompressor warning
        enable_tracking: bool = True,
        # Routing
        preferred_provider: Provider = Provider.ANTHROPIC,
        default_model: Optional[str] = None,
        enable_llm_classifier: bool = False,  # More accurate routing; costs ~15 tokens/call
        # Optimizer
        aggressive_optimization: bool = False,
        max_history_turns: int = 10,
        # Compression — only used if enable_compression=True
        max_document_tokens: int = 4000,
        compression_strategy: str = "smart",
        # Tracking
        persist_tracking: Optional[str] = None,
        # Reliability
        max_retries: int = 3,
        retry_base_delay: float = 1.0,   # seconds; doubles on each retry
        # Batching
        enable_batching: bool = False,
        batch_default_model: Optional[str] = None,
    ):
        self._anthropic = anthropic_client
        self._openai = openai_client
        self._google = google_client
        self._default_model = default_model
        self._preferred_provider = preferred_provider
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay

        self.cache_manager = CacheManager(enabled=enable_caching)

        self.router = ModelRouter(
            preferred_provider=preferred_provider,
            enable_llm_classifier=enable_llm_classifier,
            anthropic_client=anthropic_client if enable_llm_classifier else None,
        ) if enable_routing else None

        self.optimizer = PromptOptimizer(
            enabled=enable_optimization,
            aggressive=aggressive_optimization,
            max_history_turns=max_history_turns,
        )

        if enable_compression:
            warnings.warn(
                "DocumentCompressor is enabled. This feature removes content from documents "
                "to reduce token count. Answers may be incomplete or incorrect for documents "
                "where all content is potentially relevant. "
                "See DocumentCompressor docstring for guidance.",
                UserWarning,
                stacklevel=2,
            )
        self.compressor = DocumentCompressor(
            max_tokens=max_document_tokens,
            strategy=compression_strategy,
        ) if enable_compression else None

        self.tracker = CostTracker(persist_path=persist_tracking) if enable_tracking else None

        self.batcher = BatchProcessor(
            client=anthropic_client,
            default_model=batch_default_model or default_model or "claude-haiku-4-5-20251001",
        ) if enable_batching else None

    # ─── Complete (non-streaming) ──────────────────────────────────────────────

    def complete(
        self,
        messages: list[dict],
        system: Optional[str | list] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        complexity: Optional[TaskComplexity] = None,
        documents: Optional[list[str]] = None,
        force_provider: Optional[Provider] = None,
        tags: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        """
        Send a message with all optimizations applied. Blocks until response.

        For streaming responses, use client.stream() instead.

        Error behavior:
            - Retries on rate limit (429) with exponential backoff
            - Falls back to next capable model on overload errors
            - Raises the underlying provider exception after max_retries
        """
        start_time = time.time()
        optimizations: list[str] = []
        tokens_saved_total = 0

        messages, tokens_saved_total, optimizations = self._apply_pre_optimizations(
            messages, documents, optimizations, tokens_saved_total
        )

        model_key, model_config, _ = self._resolve_model(
            messages, model, complexity, force_provider, optimizations
        )

        messages, system_blocks = self.cache_manager.prepare_anthropic_messages(
            messages=messages, system=system
        )
        if self.cache_manager.enabled:
            optimizations.append("prompt_caching")

        response = self._execute_with_retry(
            model_config=model_config,
            messages=messages,
            system=system_blocks,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        latency_ms = (time.time() - start_time) * 1000
        if self.tracker and response:
            inp, out, cached = self._extract_usage(response)
            self.tracker.record(
                model_key=model_key,
                input_tokens=inp,
                output_tokens=out,
                cached_tokens=cached,
                request_type="live",
                optimizations_applied=optimizations,
                tokens_saved=tokens_saved_total,
                latency_ms=latency_ms,
                tags=tags,
            )

        return response

    # ─── Stream ───────────────────────────────────────────────────────────────

    def stream(
        self,
        messages: list[dict],
        system: Optional[str | list] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        complexity: Optional[TaskComplexity] = None,
        documents: Optional[list[str]] = None,
        force_provider: Optional[Provider] = None,
        tags: Optional[dict] = None,
        **kwargs,
    ) -> StreamResult:
        """
        Stream a response token by token. Optimizations apply identically to complete().

        Usage:
            with client.stream(messages=[...]) as stream:
                for text_chunk in stream:
                    print(text_chunk, end="", flush=True)

            # Usage stats available after stream is fully consumed
            usage = stream.usage()

        Note:
            Cost tracking records the request when stream() is called,
            with estimated token counts. Exact counts are available via stream.usage()
            after the stream completes.

        Supported providers: Anthropic, OpenAI
        Not yet supported: Google (Gemini streaming uses a different API surface)
        """
        messages, tokens_saved_total, optimizations = self._apply_pre_optimizations(
            messages, documents, [], 0
        )

        model_key, model_config, _ = self._resolve_model(
            messages, model, complexity, force_provider, optimizations
        )

        messages, system_blocks = self.cache_manager.prepare_anthropic_messages(
            messages=messages, system=system
        )

        provider = model_config.provider
        if provider == Provider.ANTHROPIC:
            if not self._anthropic:
                raise RuntimeError("No Anthropic client provided.")
            result = build_anthropic_stream(
                self._anthropic, model_config.model_id, messages,
                system_blocks, max_tokens, temperature, **kwargs
            )
        elif provider == Provider.OPENAI:
            if not self._openai:
                raise RuntimeError("No OpenAI client provided.")
            result = build_openai_stream(
                self._openai, model_config.model_id, messages,
                system_blocks, max_tokens, temperature, **kwargs
            )
        else:
            raise NotImplementedError(
                f"Streaming is not yet supported for provider: {provider.value}. "
                "Supported: anthropic, openai."
            )

        return result

    # ─── Batch ────────────────────────────────────────────────────────────────

    def queue(self, custom_id: str, messages: list[dict], system=None,
              model=None, max_tokens=1024, **kwargs) -> str:
        if not self.batcher:
            raise RuntimeError("Batching not enabled. Initialize with enable_batching=True.")
        messages, _ = self.optimizer.optimize_messages(messages)
        model_to_use = model
        if not model_to_use:
            prompt_text = " ".join(
                m.get("content", "") if isinstance(m.get("content"), str) else ""
                for m in messages
            )
            if self.router:
                model_key, _, _ = self.router.route(prompt=prompt_text)
                model_to_use = MODELS[model_key].model_id
            else:
                model_to_use = "claude-haiku-4-5-20251001"
        return self.batcher.add(
            custom_id=custom_id, messages=messages, model=model_to_use,
            system=system, max_tokens=max_tokens,
        )

    def submit_batch(self) -> Optional[str]:
        if not self.batcher:
            raise RuntimeError("Batching not enabled.")
        return self.batcher.submit()

    def poll_batch(self, batch_id: str, wait: bool = False, **kwargs) -> Any:
        if not self.batcher:
            raise RuntimeError("Batching not enabled.")
        if wait:
            return self.batcher.poll_until_complete(batch_id, **kwargs)
        return self.batcher.poll(batch_id)

    # ─── Utilities ────────────────────────────────────────────────────────────

    def estimate_cost(self, messages, system=None, model_key=None) -> dict:
        prompt_text = system or ""
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                prompt_text += content
        if not model_key and self.router:
            model_key, _, _ = self.router.route(prompt=prompt_text)
        elif not model_key:
            model_key = self._default_model or "claude-haiku-4-5"
        model = MODELS.get(model_key)
        if not model:
            return {"error": f"Unknown model: {model_key}"}
        input_tokens = estimate_tokens(prompt_text)
        est_output = int(input_tokens * 0.3)
        cost = model.estimate_cost(input_tokens, est_output)
        cached_cost = model.estimate_cost(input_tokens, est_output, cached_tokens=input_tokens)
        return {
            "model": model_key,
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": est_output,
            "estimated_cost_usd": cost,
            "cached_cost_usd": cached_cost,
            "cache_savings_usd": round(cost - cached_cost, 8),
            "pricing_warning": "Estimates use pricing.json — verify at provider pricing pages.",
        }

    def build_cached_system(self, base_instructions, few_shot_examples=None, knowledge_base=None):
        return build_cached_system_prompt(base_instructions, few_shot_examples, knowledge_base)

    def compress_document(self, document: str, query: Optional[str] = None) -> tuple[str, int]:
        if not self.compressor:
            warnings.warn(
                "compress_document() called but enable_compression=False. "
                "Initialize with enable_compression=True.",
                UserWarning, stacklevel=2,
            )
            return document, 0
        return self.compressor.compress(document, query=query)

    def summary(self) -> dict:
        if not self.tracker:
            return {"error": "Tracking disabled"}
        return self.tracker.summary()

    def print_summary(self):
        if self.tracker:
            self.tracker.print_summary()

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _apply_pre_optimizations(self, messages, documents, optimizations, tokens_saved):
        if documents and self.compressor:
            compressed_docs = []
            for doc in documents:
                compressed, saved = self.compressor.compress(doc)
                compressed_docs.append(compressed)
                tokens_saved += saved
            if tokens_saved > 0:
                optimizations.append("document_compression")
            doc_block = "\n\n---\n\n".join(compressed_docs)
            messages = [{"role": "user", "content": f"Reference documents:\n\n{doc_block}"}] + messages

        messages, prompt_saved = self.optimizer.optimize_messages(messages)
        tokens_saved += prompt_saved
        if prompt_saved > 0:
            optimizations.append("prompt_optimization")

        return messages, tokens_saved, optimizations

    def _resolve_model(self, messages, model, complexity, force_provider, optimizations):
        prompt_text = " ".join(
            m.get("content", "") if isinstance(m.get("content"), str) else ""
            for m in messages
        )
        if model:
            model_config = MODELS.get(model)
            if not model_config:
                raise ValueError(f"Unknown model key: '{model}'. Available: {list(MODELS.keys())}")
            return model, model_config, complexity or TaskComplexity.COMPLEX

        if self.router:
            model_key, model_config, detected = self.router.route(
                prompt=prompt_text,
                complexity=complexity,
                provider=force_provider or self._preferred_provider,
            )
            optimizations.append("model_routing")
            return model_key, model_config, detected

        # No router — use default or cheapest
        fallback = self._default_model or "claude-haiku-4-5"
        return fallback, MODELS[fallback], TaskComplexity.SIMPLE

    def _execute_with_retry(self, model_config, messages, system, max_tokens, temperature, **kwargs):
        """Execute with retry on rate limit and fallback on model overload."""
        last_exc = None
        delay = self._retry_base_delay

        for attempt in range(self._max_retries + 1):
            try:
                return self._execute(model_config, messages, system, max_tokens, temperature, **kwargs)
            except Exception as e:
                last_exc = e
                err_str = str(e).lower()

                is_rate_limit = "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str
                is_overload = "529" in err_str or "overload" in err_str or "capacity" in err_str

                if is_overload and attempt == 0:
                    # Try falling back to an equivalent model
                    fallback = self._get_fallback_model(model_config)
                    if fallback:
                        warnings.warn(
                            f"Model {model_config.model_id} overloaded. "
                            f"Falling back to {fallback.model_id}.",
                            UserWarning, stacklevel=3,
                        )
                        model_config = fallback
                        continue

                if (is_rate_limit or is_overload) and attempt < self._max_retries:
                    time.sleep(delay)
                    delay *= 2  # exponential backoff
                    continue

                raise  # non-retriable or out of retries

        raise last_exc  # type: ignore

    def _get_fallback_model(self, model_config: ModelConfig) -> Optional[ModelConfig]:
        """Find the next cheapest model for the same provider and complexity."""
        provider = model_config.provider
        candidates = [
            (key, m) for key, m in MODELS.items()
            if m.provider == provider
            and m.model_id != model_config.model_id
            and any(t in m.complexity_tiers for t in model_config.complexity_tiers)
        ]
        if not candidates:
            return None
        # Prefer cheaper model
        candidates.sort(key=lambda x: x[1].input_cost_per_1m)
        return candidates[0][1]

    def _execute(self, model_config, messages, system, max_tokens, temperature, **kwargs):
        provider = model_config.provider
        if provider == Provider.ANTHROPIC:
            return self._call_anthropic(model_config, messages, system, max_tokens, temperature, **kwargs)
        elif provider == Provider.OPENAI:
            return self._call_openai(model_config, messages, system, max_tokens, temperature, **kwargs)
        elif provider == Provider.GOOGLE:
            return self._call_google(model_config, messages, system, max_tokens, temperature, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _call_anthropic(self, model_config, messages, system, max_tokens, temperature, **kwargs):
        if not self._anthropic:
            raise RuntimeError(
                "No Anthropic client. Pass anthropic_client=anthropic.Anthropic() to OptimizedClient."
            )
        params: dict[str, Any] = {
            "model": model_config.model_id,
            "max_tokens": max_tokens,
            "messages": messages,
            **kwargs,
        }
        if system:
            params["system"] = system
        if temperature > 0:
            params["temperature"] = temperature
        return self._anthropic.messages.create(**params)

    def _call_openai(self, model_config, messages, system, max_tokens, temperature, **kwargs):
        if not self._openai:
            raise RuntimeError(
                "No OpenAI client. Pass openai_client=openai.OpenAI() to OptimizedClient."
            )
        all_messages = []
        if system:
            sys_text = " ".join(b.get("text", "") for b in system if b.get("type") == "text")
            all_messages.append({"role": "system", "content": sys_text})
        all_messages.extend(messages)
        return self._openai.chat.completions.create(
            model=model_config.model_id, messages=all_messages,
            max_tokens=max_tokens, temperature=temperature, **kwargs,
        )

    def _call_google(self, model_config, messages, system, max_tokens, temperature, **kwargs):
        if not self._google:
            raise RuntimeError(
                "No Google client. Pass google_client=genai.GenerativeModel() to OptimizedClient."
            )
        prompt = "\n".join(
            m.get("content", "") if isinstance(m.get("content"), str) else ""
            for m in messages
        )
        return self._google.generate_content(prompt)

    @staticmethod
    def _extract_usage(response) -> tuple[int, int, int]:
        if hasattr(response, "usage"):
            usage = response.usage
            return (
                getattr(usage, "input_tokens", 0) or 0,
                getattr(usage, "output_tokens", 0) or 0,
                getattr(usage, "cache_read_input_tokens", 0) or 0,
            )
        if hasattr(response, "usage") and hasattr(response.usage, "prompt_tokens"):
            usage = response.usage
            cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0)
            return usage.prompt_tokens, usage.completion_tokens, cached or 0
        return 0, 0, 0
