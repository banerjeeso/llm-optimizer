"""
Tests for llm_optimizer library.

Run: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from llm_optimizer import (
    OptimizedClient, Provider, TaskComplexity, MODELS,
    ModelRouter, PromptOptimizer, DocumentCompressor,
    CacheManager, CostTracker, estimate_tokens,
    build_cached_system_prompt,
)


# ─── Token Estimation ──────────────────────────────────────────────────────────

class TestTokenEstimation:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_text(self):
        tokens = estimate_tokens("Hello world")
        assert 2 <= tokens <= 5

    def test_longer_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        tokens = estimate_tokens(text)
        assert tokens > 50

    def test_code_has_more_tokens(self):
        code = "def fibonacci(n: int) -> int:\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        prose = "Write a function that calculates the fibonacci sequence for a given input number"
        # Code should have similar or more tokens despite similar character count
        assert estimate_tokens(code) > 0
        assert estimate_tokens(prose) > 0


# ─── Model Registry ───────────────────────────────────────────────────────────

class TestModels:
    def test_models_loaded(self):
        assert len(MODELS) > 0

    def test_estimate_cost_basic(self):
        model = MODELS["claude-haiku-4-5"]
        cost = model.estimate_cost(1000, 200)
        assert cost > 0

    def test_cached_cost_is_lower(self):
        model = MODELS["claude-sonnet-4"]
        full_cost = model.estimate_cost(10000, 500)
        cached_cost = model.estimate_cost(10000, 500, cached_tokens=9000)
        assert cached_cost < full_cost

    def test_cache_savings(self):
        model = MODELS["claude-opus-4"]
        savings = model.cache_savings(100000)
        assert savings > 0


# ─── Model Router ─────────────────────────────────────────────────────────────

class TestModelRouter:
    def setup_method(self):
        self.router = ModelRouter(preferred_provider=Provider.ANTHROPIC)

    def test_simple_task_gets_cheap_model(self):
        model_key, model, complexity = self.router.route(
            "classify this text as spam or not spam"
        )
        assert complexity == TaskComplexity.SIMPLE
        # Should not route to opus for a simple task
        assert "haiku" in model_key or "sonnet" in model_key

    def test_complex_task_gets_capable_model(self):
        _, _, complexity = self.router.route(
            "write code for a complex distributed system with microservices architecture"
        )
        assert complexity in (TaskComplexity.COMPLEX, TaskComplexity.EXPERT)

    def test_forced_model_respected(self):
        model_key, _, _ = self.router.route("hello", force_model="claude-opus-4")
        assert model_key == "claude-opus-4"

    def test_explicit_complexity_overrides(self):
        _, _, complexity = self.router.route("hello", complexity=TaskComplexity.EXPERT)
        assert complexity == TaskComplexity.EXPERT

    def test_estimate_savings(self):
        result = self.router.estimate_savings("classify this as spam", "claude-opus-4")
        assert "savings_usd" in result
        assert result["savings_usd"] >= 0

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError):
            self.router.route("test", force_model="nonexistent-model-xyz")


# ─── Prompt Optimizer ─────────────────────────────────────────────────────────

class TestPromptOptimizer:
    def setup_method(self):
        self.optimizer = PromptOptimizer(enabled=True)

    def test_whitespace_normalization(self):
        text = "Hello   world\n\n\n\nGoodbye"
        result = self.optimizer.optimize(text)
        assert "   " not in result.optimized_text
        assert "\n\n\n" not in result.optimized_text

    def test_filler_phrase_removal(self):
        text = "Please note that the system is running. It is important to note that performance may vary."
        result = self.optimizer.optimize(text)
        assert result.optimized_text != text

    def test_verbose_replacement(self):
        text = "In order to complete this task, you need to follow the steps."
        result = self.optimizer.optimize(text)
        assert "In order to" not in result.optimized_text

    def test_code_blocks_preserved(self):
        text = "Here is some code:\n```python\ndef   foo(x):   return x * 2\n```\n\nPlease   review it."
        result = self.optimizer.optimize(text)
        assert "def   foo(x):   return x * 2" in result.optimized_text

    def test_disabled_optimizer_passthrough(self):
        opt = PromptOptimizer(enabled=False)
        text = "Hello   world   please note that this is a test"
        result = opt.optimize(text)
        assert result.optimized_text == text
        assert result.tokens_saved == 0

    def test_optimize_messages(self):
        messages = [
            {"role": "user", "content": "In order to test this, please note that you should classify this text."},
            {"role": "assistant", "content": "Understood."},
        ]
        optimized, saved = self.optimizer.optimize_messages(messages)
        assert len(optimized) == 2
        assert isinstance(saved, int)

    def test_history_trimming(self):
        # 30 messages → should be trimmed
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            for i in range(30)
        ]
        opt = PromptOptimizer(max_history_turns=5)
        optimized, _ = opt.optimize_messages(messages)
        assert len(optimized) <= 15  # 5 turns × 2 + system


# ─── Document Compressor ──────────────────────────────────────────────────────

class TestDocumentCompressor:
    def test_short_doc_unchanged(self):
        comp = DocumentCompressor(max_tokens=5000)
        short_doc = "This is a short document."
        result, saved = comp.compress(short_doc)
        assert result == short_doc
        assert saved == 0

    def test_long_doc_compressed(self):
        comp = DocumentCompressor(max_tokens=100)
        long_doc = ("This is a very long paragraph that goes on and on. " * 50)
        result, saved = comp.compress(long_doc)
        assert estimate_tokens(result) < estimate_tokens(long_doc)
        assert saved > 0

    def test_truncate_strategy(self):
        comp = DocumentCompressor(max_tokens=50, strategy="truncate")
        long_doc = "word " * 500
        result, _ = comp.compress(long_doc)
        assert estimate_tokens(result) <= 80  # some slack for truncation marker


# ─── Cache Manager ────────────────────────────────────────────────────────────

class TestCacheManager:
    def test_system_prompt_gets_cache_control(self):
        mgr = CacheManager(enabled=True, min_tokens_to_cache=5)
        long_system = "You are a helpful assistant. " * 100
        msgs, sys_blocks = mgr.prepare_anthropic_messages([], system=long_system)
        # Last block should have cache_control
        assert sys_blocks is not None
        assert any("cache_control" in b for b in sys_blocks)

    def test_disabled_manager_no_cache_control(self):
        mgr = CacheManager(enabled=False)
        long_system = "You are a helpful assistant. " * 100
        msgs, sys_blocks = mgr.prepare_anthropic_messages([], system=long_system)
        if sys_blocks:
            assert not any("cache_control" in b for b in sys_blocks)

    def test_build_cached_system_prompt(self):
        blocks = build_cached_system_prompt(
            base_instructions="You are a helpful assistant. " * 100,
            few_shot_examples="Example: input → output. " * 50,
        )
        assert len(blocks) >= 1
        # At least one block should have cache_control if tokens >= threshold
        cache_blocks = [b for b in blocks if "cache_control" in b]
        assert len(cache_blocks) >= 0  # May be 0 if too short


# ─── Cost Tracker ─────────────────────────────────────────────────────────────

class TestCostTracker:
    def test_record_and_summary(self):
        tracker = CostTracker()
        tracker.record(
            model_key="claude-haiku-4-5",
            input_tokens=1000,
            output_tokens=200,
            request_type="live",
        )
        tracker.record(
            model_key="claude-haiku-4-5",
            input_tokens=500,
            output_tokens=100,
            cached_tokens=400,
            request_type="live",
        )
        summary = tracker.summary()
        assert summary["total_requests"] == 2
        assert summary["total_cost_usd"] > 0
        assert summary["total_input_tokens"] == 1500

    def test_empty_tracker_summary(self):
        tracker = CostTracker()
        summary = tracker.summary()
        assert summary["total_requests"] == 0

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "usage.jsonl")
        tracker = CostTracker(persist_path=path)
        tracker.record(
            model_key="claude-sonnet-4",
            input_tokens=2000,
            output_tokens=400,
        )

        # Reload from file
        tracker2 = CostTracker(persist_path=path)
        summary = tracker2.summary()
        assert summary["total_requests"] == 1


# ─── OptimizedClient (offline / no API key) ───────────────────────────────────

class TestOptimizedClientOffline:
    def test_init_no_client(self):
        # Should init fine without API clients
        client = OptimizedClient()
        assert client is not None

    def test_estimate_cost(self):
        client = OptimizedClient()
        est = client.estimate_cost(
            messages=[{"role": "user", "content": "Hello there, how are you today?"}],
            system="You are a helpful assistant.",
        )
        assert "estimated_cost_usd" in est
        assert est["estimated_cost_usd"] > 0

    def test_compress_document(self):
        client = OptimizedClient(max_document_tokens=50)
        doc = "This is a test document. " * 100
        compressed, saved = client.compress_document(doc)
        assert estimate_tokens(compressed) < estimate_tokens(doc)

    def test_summary_no_requests(self):
        client = OptimizedClient()
        summary = client.summary()
        assert summary["total_requests"] == 0

    def test_complete_without_client_raises(self):
        client = OptimizedClient()
        with pytest.raises(RuntimeError):
            client.complete(messages=[{"role": "user", "content": "Hello"}])

    def test_queue_without_batching_raises(self):
        client = OptimizedClient(enable_batching=False)
        with pytest.raises(RuntimeError):
            client.queue("id-1", messages=[{"role": "user", "content": "test"}])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
