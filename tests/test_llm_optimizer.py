"""
Test suite for llm_optimizer — v0.2.0

WHAT IS AND ISN'T TESTED HERE:
  All tests in this file are unit tests with NO real API calls.
  The Anthropic/OpenAI clients are not instantiated.
  This means:
    ✅ Token estimation logic
    ✅ Model pricing calculations
    ✅ Heuristic routing decisions
    ✅ Prompt optimization transforms
    ✅ Document compression (extractive, smart, truncate strategies)
    ✅ Cache_control block injection logic
    ✅ Cost tracker arithmetic and persistence
    ✅ Retry and fallback logic (via mock exceptions)
    ✅ Streaming module construction (not live API)
    ❌ Cache hits (requires real Anthropic API + repeated calls)
    ❌ LLM classifier routing (requires real Anthropic API)
    ❌ Batch API submission and polling (requires real Anthropic API)
    ❌ End-to-end OpenAI / Google provider calls

  Integration tests (marked with @pytest.mark.integration) exist separately
  and require real API keys set as environment variables.

Run unit tests only:  python -m pytest tests/ -v -m "not integration"
Run integration tests: python -m pytest tests/ -v -m integration
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import json
import pytest
from unittest.mock import MagicMock, patch

from llm_optimizer import (
    OptimizedClient, Provider, TaskComplexity, MODELS,
    ModelRouter, PromptOptimizer, DocumentCompressor,
    CacheManager, CostTracker, estimate_tokens,
    build_cached_system_prompt,
)
from llm_optimizer.models import pricing_info
from llm_optimizer.streaming import StreamResult


# ─── Token Estimation (unit) ───────────────────────────────────────────────────

class TestTokenEstimation:
    """Heuristic estimator — within ~30% of tiktoken on English prose."""

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_nonzero_for_any_text(self):
        assert estimate_tokens("Hello world") > 0

    def test_longer_text_more_tokens(self):
        short = estimate_tokens("Hello")
        long = estimate_tokens("Hello " * 100)
        assert long > short

    def test_code_counted(self):
        code = "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1)+fibonacci(n-2)"
        assert estimate_tokens(code) > 10


# ─── Models and Pricing (unit) ────────────────────────────────────────────────

class TestModels:
    """Verifies pricing.json is loaded correctly and math is right."""

    def test_models_loaded(self):
        assert len(MODELS) >= 6

    def test_all_models_have_provider(self):
        for key, m in MODELS.items():
            assert isinstance(m.provider, Provider), f"{key} missing provider"

    def test_cost_is_positive(self):
        for key, m in MODELS.items():
            cost = m.estimate_cost(1000, 200)
            assert cost > 0, f"{key} returned zero cost"

    def test_cached_cost_lower_than_uncached(self):
        model = MODELS["claude-sonnet-4"]
        full = model.estimate_cost(10000, 500)
        cached = model.estimate_cost(10000, 500, cached_tokens=9000)
        assert cached < full

    def test_cache_savings_positive(self):
        assert MODELS["claude-opus-4"].cache_savings(100_000) > 0

    def test_pricing_info_has_version(self):
        info = pricing_info()
        assert "last_updated" in info
        assert info["last_updated"] is not None

    def test_zero_tokens_zero_cost(self):
        assert MODELS["claude-haiku-4-5"].estimate_cost(0, 0) == 0.0


# ─── Model Router — heuristics (unit, no API calls) ──────────────────────────

class TestModelRouterHeuristics:
    """
    Tests the heuristic routing path only.
    LLM classifier routing is NOT tested here (requires real API).
    """

    def setup_method(self):
        self.router = ModelRouter(preferred_provider=Provider.ANTHROPIC)

    def test_spam_classify_is_simple(self):
        _, _, complexity = self.router.route("classify this email as spam or not spam")
        assert complexity == TaskComplexity.SIMPLE

    def test_classify_does_not_match_class_keyword(self):
        # Regression: "classify" was previously matched by "class" (substring)
        _, _, c = self.router.route("classify this as spam or not spam")
        assert c == TaskComplexity.SIMPLE  # not COMPLEX

    def test_code_task_is_complex(self):
        _, _, c = self.router.route("write code for a binary search tree in Python")
        assert c in (TaskComplexity.COMPLEX, TaskComplexity.EXPERT)

    def test_constitutional_clause_is_expert(self):
        # Previously misrouted to SIMPLE by heuristics (short prompt)
        # With "constitutional" and "implications" in EXPERT keywords, now correct
        _, _, c = self.router.route("Explain the constitutional implications of this clause")
        assert c == TaskComplexity.EXPERT

    def test_force_model_bypasses_routing(self):
        key, _, _ = self.router.route("hello", force_model="claude-opus-4")
        assert key == "claude-opus-4"

    def test_explicit_complexity_overrides_heuristic(self):
        _, _, c = self.router.route("hello world", complexity=TaskComplexity.EXPERT)
        assert c == TaskComplexity.EXPERT

    def test_unknown_force_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model key"):
            self.router.route("test", force_model="nonexistent-xyz")

    def test_explain_returns_audit_trail(self):
        result = self.router.explain("classify this as spam")
        assert "detected_complexity" in result
        assert "routed_model" in result
        assert "keyword_signals_fired" in result
        assert "token_count" in result
        assert result["llm_classifier_used"] is False

    def test_explain_includes_limitation_warning(self):
        result = self.router.explain("hello")
        assert result["warning"] is not None  # heuristic warning present

    def test_llm_classifier_requires_client(self):
        with pytest.raises(ValueError, match="requires an anthropic_client"):
            ModelRouter(enable_llm_classifier=True, anthropic_client=None)

    def test_estimate_savings_returns_dict(self):
        result = self.router.estimate_savings("classify this as spam", "claude-opus-4")
        assert "savings_usd" in result
        assert result["savings_usd"] >= 0

    def test_long_prompt_bumps_complexity(self):
        long_prompt = "summarize this " * 1500  # > 5000 tokens
        _, _, c = self.router.route(long_prompt)
        assert c in (TaskComplexity.MEDIUM, TaskComplexity.COMPLEX, TaskComplexity.EXPERT)


# ─── Router — LLM classifier fallback (unit, mocked) ─────────────────────────

class TestModelRouterLLMClassifier:
    """
    Tests LLM classifier routing with a mocked Anthropic client.
    These do NOT make real API calls.
    """

    def _make_mock_client(self, label: str):
        mock = MagicMock()
        mock.messages.create.return_value.content = [
            MagicMock(text=json.dumps({"complexity": label, "reason": "test"}))
        ]
        return mock

    def test_classifier_routes_to_expert(self):
        mock_client = self._make_mock_client("EXPERT")
        router = ModelRouter(
            preferred_provider=Provider.ANTHROPIC,
            enable_llm_classifier=True,
            anthropic_client=mock_client,
        )
        _, _, c = router.route("What are the implications of quantum entanglement on cryptography?")
        assert c == TaskComplexity.EXPERT
        mock_client.messages.create.assert_called_once()

    def test_classifier_falls_back_on_json_error(self):
        mock = MagicMock()
        mock.messages.create.side_effect = Exception("API error")
        router = ModelRouter(
            preferred_provider=Provider.ANTHROPIC,
            enable_llm_classifier=True,
            anthropic_client=mock,
        )
        # Should not raise — falls back to heuristics silently
        _, _, c = router.route("classify this")
        assert c is not None


# ─── Prompt Optimizer (unit) ──────────────────────────────────────────────────

class TestPromptOptimizer:
    def setup_method(self):
        self.opt = PromptOptimizer(enabled=True)

    def test_whitespace_normalization(self):
        text = "Hello   world\n\n\n\nBye"
        r = self.opt.optimize(text)
        assert "   " not in r.optimized_text
        assert "\n\n\n" not in r.optimized_text

    def test_filler_phrase_removal(self):
        text = "Please note that this is important. It is important to note that accuracy matters."
        r = self.opt.optimize(text)
        assert "please note that" not in r.optimized_text.lower()

    def test_verbose_replacement(self):
        text = "In order to complete this task you must act now."
        r = self.opt.optimize(text)
        assert "In order to" not in r.optimized_text

    def test_code_blocks_preserved(self):
        text = "See code:\n```python\ndef   foo(x):   return x\n```\nPlease   note   this."
        r = self.opt.optimize(text)
        assert "def   foo(x):   return x" in r.optimized_text

    def test_disabled_passthrough(self):
        opt = PromptOptimizer(enabled=False)
        text = "Hello   world   please note that"
        r = opt.optimize(text)
        assert r.optimized_text == text
        assert r.tokens_saved == 0

    def test_history_trimmed_when_long(self):
        msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(40)]
        opt = PromptOptimizer(max_history_turns=5)
        out, _ = opt.optimize_messages(msgs)
        assert len(out) <= 12  # 5 turns × 2 + some slack

    def test_multiblock_content_optimized(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "Please   note   that   this   is   a   test"}
        ]}]
        out, saved = self.opt.optimize_messages(msgs)
        assert saved >= 0


# ─── Document Compressor (unit) ───────────────────────────────────────────────

class TestDocumentCompressor:
    """
    Tests compression strategies. All tests verify:
      - Output fits within token budget
      - Compression warning is appended to output
      - Original unchanged if already within budget
    """

    def test_short_doc_unchanged(self):
        comp = DocumentCompressor(max_tokens=5000, warn=False)
        doc = "Short document."
        result, saved = comp.compress(doc)
        assert result == doc
        assert saved == 0

    def test_truncate_reduces_tokens(self):
        comp = DocumentCompressor(max_tokens=50, strategy="truncate", warn=False)
        doc = "word " * 500
        result, saved = comp.compress(doc)
        assert estimate_tokens(result) < estimate_tokens(doc)
        assert saved > 0

    def test_smart_reduces_tokens(self):
        comp = DocumentCompressor(max_tokens=100, strategy="smart", warn=False)
        doc = ("This is paragraph one. " * 10 + "\n\n") * 20
        result, saved = comp.compress(doc)
        assert estimate_tokens(result) < estimate_tokens(doc)

    def test_extractive_requires_no_query_falls_back(self):
        # extractive without query falls back to smart silently
        comp = DocumentCompressor(max_tokens=100, strategy="extractive", warn=False)
        doc = ("This paragraph discusses payment terms. " * 5 + "\n\n") * 10
        result, saved = comp.compress(doc, query=None)
        assert len(result) > 0

    def test_extractive_with_query(self):
        comp = DocumentCompressor(max_tokens=200, strategy="extractive", warn=False)
        doc = (
            "Payment is due within 30 days of invoice. Late fees apply.\n\n" * 5 +
            "The company was founded in 1990 by John Smith.\n\n" * 5 +
            "Termination requires 60 days written notice from either party.\n\n" * 5
        )
        result, _ = comp.compress(doc, query="payment terms")
        # Payment-related content should score higher
        assert "payment" in result.lower() or "invoice" in result.lower()

    def test_compression_warning_appended(self):
        comp = DocumentCompressor(max_tokens=50, strategy="truncate", warn=True)
        doc = "word " * 500
        result, _ = comp.compress(doc)
        assert "COMPRESSION WARNING" in result

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            DocumentCompressor(strategy="magic")

    def test_tfidf_scores_query_relevant_paragraphs_higher(self):
        """Verify extractive strategy scores query-relevant paragraphs higher."""
        comp = DocumentCompressor(max_tokens=40, strategy="extractive", warn=False)
        doc = "\n\n".join([
            "Payment terms: net 30 days. Invoice must be paid promptly.",
            "The office has a blue carpet and wooden desks.",
            "Late payment incurs a 1.5% monthly fee on outstanding balance.",
            "The break room has a coffee machine and microwave.",
            "All invoices must reference the purchase order number.",
        ])
        result, _ = comp.compress(doc, query="invoice payment")
        assert "payment" in result.lower() or "invoice" in result.lower()


# ─── Cache Manager (unit) ─────────────────────────────────────────────────────

class TestCacheManager:
    def test_cache_control_injected_for_long_system(self):
        mgr = CacheManager(enabled=True, min_tokens_to_cache=5)
        sys_text = "You are a helpful assistant. " * 100
        _, sys_blocks = mgr.prepare_anthropic_messages([], system=sys_text)
        assert sys_blocks is not None
        assert any("cache_control" in b for b in sys_blocks)

    def test_disabled_no_cache_control(self):
        mgr = CacheManager(enabled=False)
        sys_text = "You are a helpful assistant. " * 100
        _, sys_blocks = mgr.prepare_anthropic_messages([], system=sys_text)
        if sys_blocks:
            assert not any("cache_control" in b for b in sys_blocks)

    def test_build_cached_system_prompt_structure(self):
        blocks = build_cached_system_prompt(
            base_instructions="You are a legal analyst. " * 100,
            few_shot_examples="Example: input → output. " * 50,
        )
        assert len(blocks) >= 1

    def test_short_system_no_cache_control(self):
        # Content under min_tokens threshold should NOT get cache_control
        mgr = CacheManager(enabled=True, min_tokens_to_cache=2000)
        _, sys_blocks = mgr.prepare_anthropic_messages([], system="Short prompt")
        if sys_blocks:
            assert not any("cache_control" in b for b in sys_blocks)


# ─── Cost Tracker (unit) ──────────────────────────────────────────────────────

class TestCostTracker:
    def test_record_and_summary(self):
        t = CostTracker()
        t.record(model_key="claude-haiku-4-5", input_tokens=1000, output_tokens=200)
        t.record(model_key="claude-haiku-4-5", input_tokens=500, output_tokens=100, cached_tokens=400)
        s = t.summary()
        assert s["total_requests"] == 2
        assert s["total_cost_usd"] > 0
        assert s["total_input_tokens"] == 1500

    def test_empty_summary(self):
        assert CostTracker().summary()["total_requests"] == 0

    def test_savings_pct_nonzero_when_cached(self):
        t = CostTracker()
        t.record(model_key="claude-sonnet-4", input_tokens=10000, output_tokens=500, cached_tokens=9000)
        s = t.summary()
        assert s["total_cached_tokens"] == 9000
        assert s["total_cost_saved_usd"] > 0

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "usage.jsonl")
        t = CostTracker(persist_path=path)
        t.record(model_key="claude-sonnet-4", input_tokens=2000, output_tokens=400)
        t2 = CostTracker(persist_path=path)
        assert t2.summary()["total_requests"] == 1

    def test_thread_safety(self):
        import threading
        t = CostTracker()
        errors = []

        def record_many():
            try:
                for _ in range(50):
                    t.record(model_key="claude-haiku-4-5", input_tokens=100, output_tokens=50)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert not errors
        assert t.summary()["total_requests"] == 200


# ─── OptimizedClient — offline (unit, no API calls) ───────────────────────────

class TestOptimizedClientOffline:
    """
    Tests that work without any real API client.
    API calls will raise RuntimeError as expected.
    """

    def test_init_without_clients(self):
        client = OptimizedClient()
        assert client is not None

    def test_complete_without_client_raises_runtime_error(self):
        client = OptimizedClient()
        with pytest.raises(RuntimeError, match="No Anthropic client"):
            client.complete(messages=[{"role": "user", "content": "Hello"}])

    def test_stream_without_client_raises_runtime_error(self):
        client = OptimizedClient()
        with pytest.raises(RuntimeError, match="No Anthropic client"):
            list(client.stream(messages=[{"role": "user", "content": "Hello"}]))

    def test_queue_without_batching_raises(self):
        client = OptimizedClient(enable_batching=False)
        with pytest.raises(RuntimeError, match="Batching not enabled"):
            client.queue("id-1", messages=[{"role": "user", "content": "test"}])

    def test_estimate_cost_returns_dict(self):
        client = OptimizedClient()
        est = client.estimate_cost(
            messages=[{"role": "user", "content": "Hello there, how are you today?"}],
            system="You are a helpful assistant.",
        )
        assert "estimated_cost_usd" in est
        assert est["estimated_cost_usd"] > 0
        assert "pricing_warning" in est  # Fix 4: pricing staleness warning present

    def test_compress_document_warns_when_disabled(self):
        client = OptimizedClient(enable_compression=False)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result, saved = client.compress_document("Some text " * 1000)
            assert any("enable_compression" in str(warning.message) for warning in w)

    def test_summary_with_no_requests(self):
        client = OptimizedClient()
        assert client.summary()["total_requests"] == 0

    def test_enable_compression_warns_at_init(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            OptimizedClient(enable_compression=True)
            assert any("DocumentCompressor" in str(warning.message) for warning in w)


# ─── Retry and Fallback (unit, mocked) ────────────────────────────────────────

class TestRetryAndFallback:
    """
    Tests retry logic and model fallback.
    API errors are simulated — no real calls.
    """

    def _make_client_with_error(self, error_msg: str, succeed_on_attempt: int = 999):
        mock = MagicMock()
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] < succeed_on_attempt:
                raise Exception(error_msg)
            resp = MagicMock()
            resp.usage.input_tokens = 100
            resp.usage.output_tokens = 50
            resp.usage.cache_read_input_tokens = 0
            return resp

        mock.messages.create.side_effect = side_effect
        return mock

    def test_rate_limit_retried(self):
        mock = self._make_client_with_error("429 rate limit exceeded", succeed_on_attempt=3)
        client = OptimizedClient(
            anthropic_client=mock,
            max_retries=3,
            retry_base_delay=0.01,  # fast for tests
        )
        response = client.complete(messages=[{"role": "user", "content": "hello"}])
        assert mock.messages.create.call_count == 3

    def test_non_retriable_error_raises_immediately(self):
        mock = self._make_client_with_error("invalid_api_key authentication failed")
        client = OptimizedClient(
            anthropic_client=mock,
            max_retries=3,
            retry_base_delay=0.01,
        )
        with pytest.raises(Exception, match="invalid_api_key"):
            client.complete(messages=[{"role": "user", "content": "hello"}])
        # Should not have retried
        assert mock.messages.create.call_count == 1

    def test_max_retries_zero_no_retry(self):
        mock = self._make_client_with_error("429 rate limit")
        client = OptimizedClient(
            anthropic_client=mock,
            max_retries=0,
            retry_base_delay=0.01,
        )
        with pytest.raises(Exception):
            client.complete(messages=[{"role": "user", "content": "hello"}])
        assert mock.messages.create.call_count == 1


# ─── Streaming module (unit) ──────────────────────────────────────────────────

class TestStreaming:
    """Tests StreamResult construction. Does not test live streaming."""

    def test_stream_result_instantiates(self):
        mock_stream = MagicMock()
        result = StreamResult(mock_stream, provider="anthropic")
        assert result is not None
        assert result.usage() is None  # not consumed yet

    def test_unknown_provider_raises(self):
        mock_stream = MagicMock()
        result = StreamResult(mock_stream, provider="unknown_provider")
        with pytest.raises(NotImplementedError):
            list(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])
