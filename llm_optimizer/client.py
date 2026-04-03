"""
OptimizedClient — the main entry point for llm_optimizer.

Wraps Anthropic / OpenAI / Google clients and transparently applies:
  1. Prompt caching          (save up to 90% on repeated content)
  2. Model routing           (use cheapest capable model)
  3. Prompt optimization     (reduce token count)
  4. Batch processing        (50% savings on non-urgent requests)
  5. Document compression    (smart truncation before sending)
  6. Cost tracking           (usage reports and savings analytics)
"""

import time
from typing import Any, Optional, Union

from .models import ModelConfig, Provider, TaskComplexity, MODELS
from .caching import CacheManager, build_cached_system_prompt
from .router import ModelRouter
from .optimizer import PromptOptimizer, DocumentCompressor
from .batcher import BatchProcessor
from .tracker import CostTracker, estimate_tokens


class OptimizedClient:
    """
    Drop-in wrapper around LLM clients that automatically applies
    all cost-optimization techniques.

    Quick start:
        import anthropic
        from llm_optimizer import OptimizedClient

        raw_client = anthropic.Anthropic(api_key="sk-...")
        client = OptimizedClient(anthropic_client=raw_client)

        response = client.complete(
            messages=[{"role": "user", "content": "Hello!"}],
            system="You are a helpful assistant.",
        )
        print(response.content)
        client.tracker.print_summary()
    """

    def __init__(
        self,
        # Provider clients (pass whichever you use)
        anthropic_client=None,
        openai_client=None,
        google_client=None,
        # Feature flags
        enable_caching: bool = True,
        enable_routing: bool = True,
        enable_optimization: bool = True,
        enable_compression: bool = True,
        enable_tracking: bool = True,
        # Routing config
        preferred_provider: Provider = Provider.ANTHROPIC,
        default_model: Optional[str] = None,
        # Optimizer config
        aggressive_optimization: bool = False,
        max_history_turns: int = 10,
        # Compression config
        max_document_tokens: int = 4000,
        # Tracking config
        persist_tracking: Optional[str] = None,   # path to JSONL file
        # Batch config
        enable_batching: bool = False,
        batch_default_model: Optional[str] = None,
    ):
        self._anthropic = anthropic_client
        self._openai = openai_client
        self._google = google_client

        # Component setup
        self.cache_manager = CacheManager(enabled=enable_caching)
        self.router = ModelRouter(preferred_provider=preferred_provider)
        self.optimizer = PromptOptimizer(
            enabled=enable_optimization,
            aggressive=aggressive_optimization,
            max_history_turns=max_history_turns,
        )
        self.compressor = DocumentCompressor(
            max_tokens=max_document_tokens,
        ) if enable_compression else None

        self.tracker = CostTracker(persist_path=persist_tracking) if enable_tracking else None

        self.batcher = BatchProcessor(
            client=anthropic_client,
            default_model=batch_default_model or default_model or "claude-haiku-4-5-20251001",
        ) if enable_batching else None

        self._default_model = default_model
        self._preferred_provider = preferred_provider

    # ─── Primary API ───────────────────────────────────────────────────────────

    def complete(
        self,
        messages: list[dict],
        system: Optional[str | list] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        complexity: Optional[TaskComplexity] = None,
        documents: Optional[list[str]] = None,   # long documents to inject + compress
        force_provider: Optional[Provider] = None,
        tags: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        """
        Send a message with all optimizations applied.

        Args:
            messages:       Chat messages list
            system:         System prompt (str or Anthropic blocks list)
            model:          Model key (overrides routing). See MODELS.keys()
            max_tokens:     Max output tokens
            temperature:    Sampling temperature
            complexity:     Override auto-detected complexity tier
            documents:      Long documents to compress and inject
            force_provider: Override preferred provider
            tags:           Metadata tags for cost tracking
            **kwargs:       Passed through to the underlying client

        Returns:
            Raw response from the underlying provider client
        """
        start_time = time.time()
        optimizations: list[str] = []
        tokens_saved_total = 0

        # 1. Compress documents
        if documents and self.compressor:
            compressed_docs = []
            for doc in documents:
                compressed, saved = self.compressor.compress(doc)
                compressed_docs.append(compressed)
                tokens_saved_total += saved
            if any(tokens_saved_total > 0 for _ in compressed_docs):
                optimizations.append("document_compression")
            # Inject as first user message context
            doc_block = "\n\n---\n\n".join(compressed_docs)
            messages = [{"role": "user", "content": f"Reference documents:\n\n{doc_block}"}] + messages

        # 2. Optimize prompts
        messages, prompt_tokens_saved = self.optimizer.optimize_messages(messages)
        tokens_saved_total += prompt_tokens_saved
        if prompt_tokens_saved > 0:
            optimizations.append("prompt_optimization")

        # 3. Route to optimal model
        prompt_text = " ".join(
            m.get("content", "") if isinstance(m.get("content"), str) else ""
            for m in messages
        )
        if model:
            model_key = model
            model_config = MODELS.get(model_key)
            detected_complexity = complexity or TaskComplexity.COMPLEX
        else:
            model_key, model_config, detected_complexity = self.router.route(
                prompt=prompt_text,
                complexity=complexity,
                provider=force_provider or self._preferred_provider,
            )
            if model_key != (self._default_model or model_key):
                optimizations.append("model_routing")

        if not model_config:
            raise ValueError(f"Unknown model key: {model_key}")

        # 4. Apply caching
        messages, system_blocks = self.cache_manager.prepare_anthropic_messages(
            messages=messages,
            system=system,
        )
        if self.cache_manager.enabled:
            optimizations.append("prompt_caching")

        # 5. Execute request
        response = self._execute(
            model_config=model_config,
            messages=messages,
            system=system_blocks,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        # 6. Track usage
        latency_ms = (time.time() - start_time) * 1000
        if self.tracker and response:
            input_tokens, output_tokens, cached_tokens = self._extract_usage(response)
            self.tracker.record(
                model_key=model_key,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                request_type="live",
                optimizations_applied=optimizations,
                tokens_saved=tokens_saved_total,
                latency_ms=latency_ms,
                tags=tags,
            )

        return response

    def queue(
        self,
        custom_id: str,
        messages: list[dict],
        system: Optional[str | list] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """
        Queue a request for batch processing (50% cost savings).
        Requires enable_batching=True.

        Returns:
            custom_id for tracking the result later
        """
        if not self.batcher:
            raise RuntimeError(
                "Batching is disabled. Initialize with enable_batching=True."
            )

        # Still apply prompt optimization before queuing
        messages, _ = self.optimizer.optimize_messages(messages)

        model_to_use = model
        if not model_to_use:
            prompt_text = " ".join(
                m.get("content", "") if isinstance(m.get("content"), str) else ""
                for m in messages
            )
            model_key, _, _ = self.router.route(prompt=prompt_text)
            model_to_use = MODELS[model_key].model_id

        return self.batcher.add(
            custom_id=custom_id,
            messages=messages,
            model=model_to_use,
            system=system,
            max_tokens=max_tokens,
        )

    def submit_batch(self) -> Optional[str]:
        """Submit queued requests as a batch. Returns batch_id."""
        if not self.batcher:
            raise RuntimeError("Batching not enabled.")
        return self.batcher.submit()

    def poll_batch(self, batch_id: str, wait: bool = False, **kwargs) -> Any:
        """Poll a submitted batch. If wait=True, blocks until complete."""
        if not self.batcher:
            raise RuntimeError("Batching not enabled.")
        if wait:
            return self.batcher.poll_until_complete(batch_id, **kwargs)
        return self.batcher.poll(batch_id)

    # ─── Convenience Methods ───────────────────────────────────────────────────

    def estimate_cost(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        model_key: Optional[str] = None,
    ) -> dict:
        """
        Estimate cost before sending a request.
        Useful for gating expensive calls.
        """
        prompt_text = system or ""
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                prompt_text += content

        if not model_key:
            model_key, _, _ = self.router.route(prompt=prompt_text)

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
        }

    def build_cached_system(
        self,
        base_instructions: str,
        few_shot_examples: Optional[str] = None,
        knowledge_base: Optional[str] = None,
    ) -> list[dict]:
        """Build an optimally cached system prompt structure."""
        return build_cached_system_prompt(base_instructions, few_shot_examples, knowledge_base)

    def compress_document(self, document: str, query: Optional[str] = None) -> tuple[str, int]:
        """Compress a document to fit within token budget."""
        if not self.compressor:
            return document, 0
        return self.compressor.compress(document, query=query)

    def summary(self) -> dict:
        """Return usage and cost summary."""
        if not self.tracker:
            return {"error": "Tracking disabled"}
        return self.tracker.summary()

    def print_summary(self):
        """Print a formatted usage/cost report."""
        if self.tracker:
            self.tracker.print_summary()

    # ─── Internal Execution ────────────────────────────────────────────────────

    def _execute(
        self,
        model_config: ModelConfig,
        messages: list[dict],
        system: Optional[list],
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Any:
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
            raise RuntimeError("No Anthropic client provided. Pass anthropic_client=anthropic.Anthropic()")

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
            raise RuntimeError("No OpenAI client provided. Pass openai_client=openai.OpenAI()")

        all_messages = []
        if system:
            sys_text = " ".join(b.get("text", "") for b in system if b.get("type") == "text")
            all_messages.append({"role": "system", "content": sys_text})
        all_messages.extend(messages)

        return self._openai.chat.completions.create(
            model=model_config.model_id,
            messages=all_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def _call_google(self, model_config, messages, system, max_tokens, temperature, **kwargs):
        if not self._google:
            raise RuntimeError("No Google client provided. Pass google_client=genai.GenerativeModel(...)")
        # Google Generative AI SDK call
        prompt = "\n".join(
            m.get("content", "") if isinstance(m.get("content"), str) else ""
            for m in messages
        )
        return self._google.generate_content(prompt)

    @staticmethod
    def _extract_usage(response: Any) -> tuple[int, int, int]:
        """Extract (input_tokens, output_tokens, cached_tokens) from response."""
        # Anthropic
        if hasattr(response, "usage"):
            usage = response.usage
            input_t = getattr(usage, "input_tokens", 0) or 0
            output_t = getattr(usage, "output_tokens", 0) or 0
            cached_t = getattr(usage, "cache_read_input_tokens", 0) or 0
            return input_t, output_t, cached_t
        # OpenAI
        if hasattr(response, "usage") and hasattr(response.usage, "prompt_tokens"):
            usage = response.usage
            cached = getattr(usage, "prompt_tokens_details", None)
            cached_t = getattr(cached, "cached_tokens", 0) if cached else 0
            return usage.prompt_tokens, usage.completion_tokens, cached_t
        return 0, 0, 0
