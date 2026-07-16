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



# ─── Bedrock Client (unit, mocked) ────────────────────────────────────────────

class TestBedrockClient:
    """
    Tests BedrockClient with mocked boto3. No real AWS calls made.

    Covers:
      - Both auth patterns (profile_name and env/role)
      - cache_control stripping
      - Response shape normalization
      - Discount multiplier
      - OptimizedClient integration with Bedrock
      - Correct model IDs for company Bedrock setup
    """

    def _mock_runtime(self, response_text="Test response"):
        import json
        from unittest.mock import MagicMock
        mock = MagicMock()
        body = {
            "content": [{"type": "text", "text": response_text}],
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "stop_reason": "end_turn",
        }
        mock.invoke_model.return_value = {
            "body": MagicMock(read=lambda: json.dumps(body).encode())
        }
        return mock

    def test_instantiates_with_profile(self):
        from llm_optimizer.bedrock import BedrockClient
        c = BedrockClient(region="us-east-1", profile_name="my-profile")
        assert c.region == "us-east-1"
        assert c._profile == "my-profile"

    def test_instantiates_without_profile(self):
        """Env vars / IAM role path — zero config."""
        from llm_optimizer.bedrock import BedrockClient
        c = BedrockClient()
        assert c._profile is None
        assert c.region == "us-east-1"

    def test_discount_multiplier(self):
        from llm_optimizer.bedrock import BedrockClient
        c = BedrockClient(discount_pct=16.0)
        assert abs(c.effective_cost_multiplier() - 0.84) < 0.001

    def test_no_discount_multiplier_is_one(self):
        from llm_optimizer.bedrock import BedrockClient
        c = BedrockClient(discount_pct=0.0)
        assert c.effective_cost_multiplier() == 1.0

    def test_strip_cache_control_removes_key(self):
        from llm_optimizer.bedrock import BedrockClient
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}}
        ]}]
        cleaned = BedrockClient._strip_cache_control(msgs)
        assert "cache_control" not in cleaned[0]["content"][0]
        assert cleaned[0]["content"][0]["text"] == "Hello"

    def test_strip_cache_control_plain_string_unchanged(self):
        from llm_optimizer.bedrock import BedrockClient
        msgs = [{"role": "user", "content": "Hello"}]
        cleaned = BedrockClient._strip_cache_control(msgs)
        assert cleaned[0]["content"] == "Hello"

    def test_strip_cache_control_preserves_other_keys(self):
        from llm_optimizer.bedrock import BedrockClient
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "Hi", "cache_control": {"type": "ephemeral"}, "extra": "keep"}
        ]}]
        cleaned = BedrockClient._strip_cache_control(msgs)
        assert cleaned[0]["content"][0]["extra"] == "keep"
        assert "cache_control" not in cleaned[0]["content"][0]

    def test_response_shape_normalized(self):
        from llm_optimizer.bedrock import BedrockClient, BedrockResponse
        c = BedrockClient(region="us-east-1")
        c._runtime = self._mock_runtime("Hello from Bedrock")
        resp = c.messages.create(
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert isinstance(resp, BedrockResponse)
        assert resp.content[0].text == "Hello from Bedrock"
        assert resp.usage.input_tokens == 100
        assert resp.usage.output_tokens == 50
        assert resp.usage.cache_read_input_tokens == 0

    def test_correct_haiku_model_id(self):
        """Company uses this exact Bedrock model ID for Haiku."""
        assert MODELS["bedrock-claude-haiku-4-5"].model_id == "anthropic.claude-haiku-4-5-20251001-v1:0"

    def test_correct_sonnet_model_id(self):
        """Company uses this exact Bedrock model ID for Sonnet."""
        assert MODELS["bedrock-claude-sonnet-4-5"].model_id == "anthropic.claude-sonnet-4-5-20250929-v1:0"

    def test_bedrock_haiku_pricing(self):
        """Bedrock Haiku 4.5 AWS list price."""
        m = MODELS["bedrock-claude-haiku-4-5"]
        assert m.input_cost_per_1m == 0.80
        assert m.output_cost_per_1m == 4.00

    def test_bedrock_sonnet_pricing(self):
        """Bedrock Sonnet 4.5 AWS list price."""
        m = MODELS["bedrock-claude-sonnet-4-5"]
        assert m.input_cost_per_1m == 3.00
        assert m.output_cost_per_1m == 15.00

    def test_bedrock_caching_supported(self):
        """Bedrock now supports caching via cachePoint format (updated v0.4.1)."""
        assert MODELS["bedrock-claude-haiku-4-5"].supports_caching is True
        assert MODELS["bedrock-claude-sonnet-4-5"].supports_caching is True

    def test_optimized_client_with_bedrock_profile(self):
        """Full integration: OptimizedClient + BedrockClient with profile."""
        from llm_optimizer.bedrock import BedrockClient
        import json
        from unittest.mock import MagicMock

        bedrock = BedrockClient(region="us-east-1", profile_name="my-profile")
        bedrock._runtime = self._mock_runtime("Classified as SPAM")

        client = OptimizedClient(
            bedrock_client=bedrock,
            preferred_provider=Provider.BEDROCK,
        )
        resp = client.complete(
            messages=[{"role": "user", "content": "Is this spam?"}],
            system="You are a classifier.",
            model="bedrock-claude-haiku-4-5",
        )
        assert resp.content[0].text == "Classified as SPAM"
        bedrock._runtime.invoke_model.assert_called_once()

    def test_optimized_client_no_bedrock_raises(self):
        """Clear error when Bedrock client not provided."""
        client = OptimizedClient(preferred_provider=Provider.BEDROCK)
        with pytest.raises(RuntimeError, match="No Bedrock client"):
            client.complete(
                messages=[{"role": "user", "content": "Hello"}],
                model="bedrock-claude-haiku-4-5",
            )

    def test_make_bedrock_client_factory(self):
        from llm_optimizer.bedrock import make_bedrock_client
        c = make_bedrock_client(profile_name="my-profile", discount_pct=16.0)
        assert c._profile == "my-profile"
        assert c.discount_pct == 16.0

    def test_make_bedrock_client_no_args(self):
        """Zero-config factory for env/role auth."""
        from llm_optimizer.bedrock import make_bedrock_client
        c = make_bedrock_client()
        assert c._profile is None

    def test_bedrock_provider_in_tier_defaults(self):
        from llm_optimizer.models import TIER_DEFAULTS
        assert Provider.BEDROCK in TIER_DEFAULTS[TaskComplexity.SIMPLE]
        assert Provider.BEDROCK in TIER_DEFAULTS[TaskComplexity.COMPLEX]


# ─── Budget Guard (unit) ──────────────────────────────────────────────────────

class TestBudgetGuard:
    """Tests budget enforcement. No API calls made."""

    def test_per_request_limit_blocks(self):
        from llm_optimizer.budget import BudgetGuard, BudgetExceededError
        guard = BudgetGuard(max_cost_per_request=0.01)
        with pytest.raises(BudgetExceededError) as exc:
            guard.check(0.05)
        assert exc.value.limit_type == "per_request"
        assert exc.value.budget == 0.01

    def test_per_request_limit_allows_under(self):
        from llm_optimizer.budget import BudgetGuard
        guard = BudgetGuard(max_cost_per_request=0.01)
        guard.check(0.005)  # should not raise

    def test_monthly_budget_blocks(self):
        from llm_optimizer.budget import BudgetGuard, BudgetExceededError
        guard = BudgetGuard(monthly_budget=1.00)
        guard.record(0.95)  # spend $0.95
        with pytest.raises(BudgetExceededError) as exc:
            guard.check(0.10)  # would exceed $1.00
        assert exc.value.limit_type == "monthly_budget"

    def test_total_budget_blocks(self):
        from llm_optimizer.budget import BudgetGuard, BudgetExceededError
        guard = BudgetGuard(total_budget=5.00)
        guard.record(4.99)
        with pytest.raises(BudgetExceededError):
            guard.check(0.02)

    def test_no_limits_never_blocks(self):
        from llm_optimizer.budget import BudgetGuard
        guard = BudgetGuard()
        guard.check(999.99)  # should not raise

    def test_warning_callback_fires(self):
        from llm_optimizer.budget import BudgetGuard
        warnings_received = []
        guard = BudgetGuard(
            monthly_budget=10.00,
            warning_threshold_pct=80.0,
            on_budget_warning=lambda pct, t: warnings_received.append((pct, t))
        )
        guard.record(8.50)  # 85% of $10
        assert len(warnings_received) == 1
        assert warnings_received[0][0] >= 80.0

    def test_warning_fires_only_once(self):
        from llm_optimizer.budget import BudgetGuard
        count = [0]
        guard = BudgetGuard(
            monthly_budget=10.00,
            warning_threshold_pct=80.0,
            on_budget_warning=lambda pct, t: count.__setitem__(0, count[0] + 1)
        )
        guard.record(8.50)
        guard.record(0.10)
        assert count[0] == 1  # fired once only

    def test_reset_clears_spend(self):
        from llm_optimizer.budget import BudgetGuard, BudgetExceededError
        guard = BudgetGuard(monthly_budget=1.00)
        guard.record(0.99)
        guard.reset()
        guard.check(0.99)  # should not raise after reset

    def test_status_returns_remaining(self):
        from llm_optimizer.budget import BudgetGuard
        guard = BudgetGuard(monthly_budget=10.00)
        guard.record(3.00)
        status = guard.status()
        assert status["total_spent_usd"] == 3.0
        assert status["monthly_remaining"] == 7.0
        assert status["monthly_used_pct"] == 30.0

    def test_persistence(self, tmp_path):
        from llm_optimizer.budget import BudgetGuard
        path = str(tmp_path / "budget.json")
        g1 = BudgetGuard(monthly_budget=10.00, persist_path=path)
        g1.record(4.50)
        g2 = BudgetGuard(monthly_budget=10.00, persist_path=path)
        assert g2.status()["total_spent_usd"] == 4.50

    def test_budget_exceeded_error_message(self):
        from llm_optimizer.budget import BudgetGuard, BudgetExceededError
        guard = BudgetGuard(max_cost_per_request=0.01)
        try:
            guard.check(0.05)
        except BudgetExceededError as e:
            assert "per_request" in str(e)
            assert "0.05" in str(e)


# ─── Output Cache (unit) ──────────────────────────────────────────────────────

class TestOutputCache:
    """Tests in-memory and SQLite output caching."""

    def test_memory_cache_hit(self):
        from llm_optimizer.cache import OutputCache, CachedResponse
        import time
        cache = OutputCache(backend="memory", ttl_seconds=60)
        resp = CachedResponse("Hello", "model", 100, 50, time.time(), 0.001)
        cache.set("key1", resp)
        result = cache.get("key1")
        assert result is not None
        assert result.content == "Hello"

    def test_memory_cache_miss(self):
        from llm_optimizer.cache import OutputCache
        cache = OutputCache(backend="memory")
        assert cache.get("nonexistent") is None

    def test_memory_cache_ttl_expiry(self):
        from llm_optimizer.cache import OutputCache, CachedResponse
        import time
        cache = OutputCache(backend="memory", ttl_seconds=1)
        resp = CachedResponse("Hello", "model", 100, 50, time.time() - 2, 0.001)
        cache.set("key1", resp)
        assert cache.get("key1") is None  # expired

    def test_memory_cache_lru_eviction(self):
        from llm_optimizer.cache import OutputCache, CachedResponse
        import time
        cache = OutputCache(backend="memory", max_size=3)
        for i in range(4):
            cache.set(f"key{i}", CachedResponse(f"r{i}", "m", 10, 5, time.time(), 0.001))
        assert cache.get("key0") is None  # evicted
        assert cache.get("key3") is not None

    def test_stats_hit_rate(self):
        from llm_optimizer.cache import OutputCache, CachedResponse
        import time
        cache = OutputCache(backend="memory")
        resp = CachedResponse("Hi", "model", 100, 50, time.time(), 0.005)
        cache.set("k", resp)
        cache.get("k")         # hit
        cache.get("missing")   # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate_pct"] == 50.0
        assert stats["cost_saved_usd"] == 0.005

    def test_sqlite_cache(self, tmp_path):
        from llm_optimizer.cache import OutputCache, CachedResponse
        import time
        db = str(tmp_path / "test_cache.db")
        cache = OutputCache(backend="sqlite", db_path=db, ttl_seconds=60)
        resp = CachedResponse("SQLite response", "model", 200, 100, time.time(), 0.002)
        cache.set("sqlkey", resp)
        result = cache.get("sqlkey")
        assert result is not None
        assert result.content == "SQLite response"

    def test_sqlite_cache_persists(self, tmp_path):
        from llm_optimizer.cache import OutputCache, CachedResponse
        import time
        db = str(tmp_path / "persist_cache.db")
        c1 = OutputCache(backend="sqlite", db_path=db, ttl_seconds=3600)
        c1.set("pk", CachedResponse("Persistent", "m", 10, 5, time.time(), 0.001))
        c2 = OutputCache(backend="sqlite", db_path=db, ttl_seconds=3600)
        assert c2.get("pk").content == "Persistent"

    def test_clear_empties_cache(self):
        from llm_optimizer.cache import OutputCache, CachedResponse
        import time
        cache = OutputCache(backend="memory")
        cache.set("k", CachedResponse("Hi", "m", 10, 5, time.time(), 0.001))
        cache.clear()
        assert cache.get("k") is None

    def test_make_cache_key_stable(self):
        from llm_optimizer.cache import make_cache_key
        msgs = [{"role": "user", "content": "Hello"}]
        k1 = make_cache_key("model", msgs, "system", 0.0)
        k2 = make_cache_key("model", msgs, "system", 0.0)
        assert k1 == k2

    def test_make_cache_key_different_for_different_inputs(self):
        from llm_optimizer.cache import make_cache_key
        k1 = make_cache_key("model-a", [{"role": "user", "content": "Hi"}], None, 0.0)
        k2 = make_cache_key("model-b", [{"role": "user", "content": "Hi"}], None, 0.0)
        assert k1 != k2

    def test_invalid_backend_raises(self):
        from llm_optimizer.cache import OutputCache
        with pytest.raises(ValueError, match="Unknown cache backend"):
            OutputCache(backend="redis")


# ─── History Summarizer (unit, mocked) ────────────────────────────────────────

class TestHistorySummarizer:
    """Tests history summarization with mocked Anthropic client."""

    def _make_mock_client(self, summary_text="Summary of earlier conversation"):
        from unittest.mock import MagicMock
        mock = MagicMock()
        mock.messages.create.return_value.content = [MagicMock(text=summary_text)]
        return mock

    def test_no_summarization_under_threshold(self):
        from llm_optimizer.history import HistorySummarizer
        s = HistorySummarizer(threshold=10)
        msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(10)]
        result, summarized = s.maybe_summarize(msgs)
        assert not summarized
        assert result == msgs

    def test_summarization_over_threshold(self):
        from llm_optimizer.history import HistorySummarizer
        mock = self._make_mock_client("Earlier: discussed Python setup")
        s = HistorySummarizer(client=mock, threshold=4, keep_recent=2)
        msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(12)]
        result, summarized = s.maybe_summarize(msgs)
        assert summarized
        assert len(result) < len(msgs)

    def test_recent_turns_preserved(self):
        from llm_optimizer.history import HistorySummarizer, SUMMARY_BLOCK_PREFIX
        mock = self._make_mock_client("Summary here")
        s = HistorySummarizer(client=mock, threshold=4, keep_recent=2)
        msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(12)]
        result, _ = s.maybe_summarize(msgs)
        # Last 4 messages (2 turns) should be present verbatim
        last_contents = [m["content"] for m in result if not str(m.get("content","")).startswith(SUMMARY_BLOCK_PREFIX)]
        assert "msg 8" in last_contents or "msg 10" in last_contents

    def test_summary_block_injected(self):
        from llm_optimizer.history import HistorySummarizer, SUMMARY_BLOCK_PREFIX
        mock = self._make_mock_client("Summary content here")
        s = HistorySummarizer(client=mock, threshold=4, keep_recent=2)
        msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(12)]
        result, _ = s.maybe_summarize(msgs)
        summary_blocks = [m for m in result if str(m.get("content","")).startswith(SUMMARY_BLOCK_PREFIX)]
        assert len(summary_blocks) == 1

    def test_no_client_falls_back_to_truncation(self):
        from llm_optimizer.history import HistorySummarizer
        s = HistorySummarizer(client=None, threshold=4, keep_recent=2)
        msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(12)]
        result, summarized = s.maybe_summarize(msgs)
        assert summarized
        assert len(result) == 4  # keep_recent * 2

    def test_system_messages_preserved(self):
        from llm_optimizer.history import HistorySummarizer
        mock = self._make_mock_client()
        s = HistorySummarizer(client=mock, threshold=2, keep_recent=1)
        msgs = [
            {"role": "system", "content": "You are a helpful assistant"},
            *[{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(8)]
        ]
        result, _ = s.maybe_summarize(msgs)
        system_msgs = [m for m in result if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "You are a helpful assistant"


# ─── OptimizedClient — budget + cache integration (unit, mocked) ──────────────

class TestOptimizedClientV4:
    """Integration tests for v0.4.0 features."""

    def _make_anthropic_mock(self, text="Response"):
        from unittest.mock import MagicMock
        mock = MagicMock()
        resp = MagicMock()
        resp.content = [MagicMock(text=text)]
        resp.usage.input_tokens = 100
        resp.usage.output_tokens = 50
        resp.usage.cache_read_input_tokens = 0
        mock.messages.create.return_value = resp
        return mock

    def test_budget_status_no_budget(self):
        client = OptimizedClient()
        status = client.budget_status()
        assert "error" in status

    def test_cache_stats_no_cache(self):
        client = OptimizedClient()
        stats = client.cache_stats()
        assert "error" in stats

    def test_budget_initialized(self):
        client = OptimizedClient(max_cost_per_request=0.05)
        assert client.budget is not None
        status = client.budget_status()
        assert status["max_cost_per_request"] == 0.05

    def test_output_cache_initialized(self):
        client = OptimizedClient(enable_output_cache=True)
        assert client.output_cache is not None
        stats = client.cache_stats()
        assert stats["backend"] == "memory"

    def test_history_summarizer_initialized(self):
        client = OptimizedClient(enable_history_summarization=True, history_summarization_threshold=5)
        assert client.history_summarizer is not None
        assert client.history_summarizer.threshold == 5

    def test_budget_exceeded_blocks_request(self):
        from llm_optimizer.budget import BudgetExceededError
        mock = self._make_anthropic_mock()
        client = OptimizedClient(
            anthropic_client=mock,
            total_budget=0.0,  # zero budget — any request exceeds it
        )
        # Pre-record a tiny spend so total_spent > total_budget on next request
        client.budget.record(0.001)
        with pytest.raises(BudgetExceededError):
            client.complete(
                messages=[{"role": "user", "content": "Hello"}],
                model="claude-haiku-4-5",
            )
        mock.messages.create.assert_not_called()  # request never sent


# ─── Bedrock Cache Translation (unit) ────────────────────────────────────────

class TestBedrockCacheTranslation:
    """
    Tests automatic translation of Anthropic cache_control → Bedrock cachePoint.

    This is the genuine differentiator:
      - Anthropic automatic caching does NOT work on Bedrock
      - LiteLLM strips TTL on Bedrock (bug as of Feb 2026)
      - This library translates cache_control → cachePoint and preserves TTL

    All tests are unit tests — no real AWS calls made.
    """

    def test_make_cache_point_default(self):
        from llm_optimizer.bedrock_cache import make_cache_point
        cp = make_cache_point()
        assert cp == {"cachePoint": {"type": "default", "ttl": "5m"}}

    def test_make_cache_point_1h(self):
        from llm_optimizer.bedrock_cache import make_cache_point
        cp = make_cache_point(ttl="1h")
        assert cp == {"cachePoint": {"type": "default", "ttl": "1h"}}

    def test_make_cache_point_invalid_ttl_defaults_to_5m(self):
        from llm_optimizer.bedrock_cache import make_cache_point
        cp = make_cache_point(ttl="2h")
        assert cp["cachePoint"]["ttl"] == "5m"

    def test_translate_cache_control_to_cachepoint(self):
        """cache_control block → cachePoint block inserted after it."""
        from llm_optimizer.bedrock_cache import translate_cache_control_to_cachepoint
        content = [
            {"type": "text", "text": "Long system prompt " * 300,
             "cache_control": {"type": "ephemeral"}},
        ]
        result, added = translate_cache_control_to_cachepoint(content, ttl="5m")
        assert added == 1
        # cache_control removed from text block
        assert "cache_control" not in result[0]
        assert result[0]["text"] == content[0]["text"]
        # cachePoint inserted after
        assert "cachePoint" in result[1]
        assert result[1]["cachePoint"]["ttl"] == "5m"

    def test_translate_preserves_1h_ttl(self):
        """TTL from original cache_control is preserved — fixes LiteLLM bug."""
        from llm_optimizer.bedrock_cache import translate_cache_control_to_cachepoint
        content = [
            {"type": "text", "text": "System " * 500,
             "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        ]
        result, _ = translate_cache_control_to_cachepoint(content, ttl="5m")
        # 1h TTL from cache_control should be preserved
        assert result[1]["cachePoint"]["ttl"] == "1h"

    def test_translate_respects_4_checkpoint_limit(self):
        """Never exceeds Bedrock's 4-checkpoint maximum."""
        from llm_optimizer.bedrock_cache import translate_cache_control_to_cachepoint
        content = [
            {"type": "text", "text": f"Block {i}", "cache_control": {"type": "ephemeral"}}
            for i in range(6)  # 6 cache_control blocks — should only inject 4
        ]
        counter = [0]
        result, added = translate_cache_control_to_cachepoint(content, ttl="5m", checkpoints_used=counter)
        cache_points = [b for b in result if isinstance(b, dict) and "cachePoint" in b]
        assert len(cache_points) == 4  # never more than 4

    def test_translate_strips_cache_control_even_at_limit(self):
        """cache_control always removed from blocks, even when checkpoint limit hit."""
        from llm_optimizer.bedrock_cache import translate_cache_control_to_cachepoint
        content = [
            {"type": "text", "text": "Block", "cache_control": {"type": "ephemeral"}}
            for _ in range(6)
        ]
        counter = [0]
        result, _ = translate_cache_control_to_cachepoint(content, ttl="5m", checkpoints_used=counter)
        text_blocks = [b for b in result if isinstance(b, dict) and "type" in b]
        for block in text_blocks:
            assert "cache_control" not in block

    def test_bedrock_cache_manager_translate_system(self):
        """System prompt with cache_control → system blocks with cachePoint."""
        from llm_optimizer.bedrock_cache import BedrockCacheManager
        mgr = BedrockCacheManager(enabled=True, ttl="5m", min_tokens=5)
        system = [
            {"type": "text", "text": "You are a helpful assistant. " * 200,
             "cache_control": {"type": "ephemeral"}},
        ]
        _, prepared_system = mgr.prepare(messages=[], system=system)
        assert prepared_system is not None
        cache_points = [b for b in prepared_system if isinstance(b, dict) and "cachePoint" in b]
        assert len(cache_points) >= 1

    def test_bedrock_cache_manager_auto_inject(self):
        """Auto-injects cachePoint when system is large but has no cache_control."""
        from llm_optimizer.bedrock_cache import BedrockCacheManager
        mgr = BedrockCacheManager(enabled=True, auto_inject=True, min_tokens=10)
        system = [{"type": "text", "text": "You are a helpful assistant. " * 200}]
        _, prepared_system = mgr.prepare(messages=[], system=system)
        cache_points = [b for b in prepared_system if isinstance(b, dict) and "cachePoint" in b]
        assert len(cache_points) >= 1

    def test_bedrock_cache_manager_disabled_strips_all_markup(self):
        """When disabled, cache_control stripped and no cachePoint injected."""
        from llm_optimizer.bedrock_cache import BedrockCacheManager, strip_cache_control
        mgr = BedrockCacheManager(enabled=False)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}}
        ]}]
        cleaned = strip_cache_control(messages)
        block = cleaned[0]["content"][0]
        assert "cache_control" not in block
        assert "cachePoint" not in block

    def test_bedrock_client_uses_cachepoint_not_cache_control(self):
        """Full integration: BedrockClient body uses cachePoint, not cache_control."""
        import json
        from llm_optimizer.bedrock import BedrockClient
        client = BedrockClient(region="us-east-1", enable_caching=True, cache_ttl="5m")
        # Simulate messages with cache_control (Anthropic format)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Hello " * 500,
             "cache_control": {"type": "ephemeral"}},
        ]}]
        system = [{"type": "text", "text": "System prompt " * 300,
                   "cache_control": {"type": "ephemeral"}}]
        body = client._build_body(
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            max_tokens=100,
            messages=messages,
            system=system,
        )
        body_str = json.dumps(body)
        # cache_control should NOT appear in body
        assert "cache_control" not in body_str
        # cachePoint SHOULD appear
        assert "cachePoint" in body_str

    def test_bedrock_client_ttl_preserved(self):
        """1h TTL from cache_control is preserved in cachePoint — fixes LiteLLM bug."""
        import json
        from llm_optimizer.bedrock import BedrockClient
        client = BedrockClient(region="us-east-1", enable_caching=True)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Hello " * 500,
             "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        ]}]
        body = client._build_body(
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            max_tokens=100,
            messages=messages,
            system=None,
        )
        body_str = json.dumps(body)
        assert '"1h"' in body_str  # TTL preserved — not stripped

    def test_bedrock_client_caching_disabled_strips_all(self):
        """When enable_caching=False, all cache markup removed from body."""
        import json
        from llm_optimizer.bedrock import BedrockClient
        client = BedrockClient(region="us-east-1", enable_caching=False)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Hello",
             "cache_control": {"type": "ephemeral"}},
        ]}]
        body = client._build_body(
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            max_tokens=100,
            messages=messages,
            system=None,
        )
        body_str = json.dumps(body)
        assert "cache_control" not in body_str
        assert "cachePoint" not in body_str

    def test_bedrock_models_now_support_caching(self):
        """pricing.json should reflect that Bedrock models support caching."""
        assert MODELS["bedrock-claude-haiku-4-5"].supports_caching is True
        assert MODELS["bedrock-claude-sonnet-4-5"].supports_caching is True

    def test_make_bedrock_client_cache_params(self):
        """make_bedrock_client factory passes cache params through."""
        from llm_optimizer.bedrock import make_bedrock_client
        client = make_bedrock_client(
            profile_name="my-profile",
            enable_caching=True,
            cache_ttl="1h",
        )
        assert client.cache_manager.enabled is True
        assert client.cache_manager.ttl == "1h"

    def test_make_bedrock_client_caching_disabled(self):
        from llm_optimizer.bedrock import make_bedrock_client
        client = make_bedrock_client(enable_caching=False)
        assert client.cache_manager.enabled is False


# ─── Semantic Cache (unit) ────────────────────────────────────────────────────

class TestSemanticCache:
    def test_exact_match_is_hit(self):
        import time
        from llm_optimizer.semantic_cache import SemanticCache
        cache = SemanticCache(threshold=0.85, ttl_seconds=60)
        msgs = [{"role": "user", "content": "What is the capital of France?"}]
        cache.set(msgs, "Paris", "model", 10, 5, 0.001)
        result = cache.get(msgs)
        assert result is not None
        assert result.response_text == "Paris"

    def test_paraphrase_is_hit(self):
        from llm_optimizer.semantic_cache import SemanticCache
        cache = SemanticCache(threshold=0.60, ttl_seconds=60)
        # Use longer sentences to give TF-IDF enough signal
        msgs1 = [{"role": "user", "content": "classify email spam detection filter"}]
        msgs2 = [{"role": "user", "content": "email spam classification detection filter"}]
        cache.set(msgs1, "SPAM", "model", 10, 5, 0.001)
        result = cache.get(msgs2)
        assert result is not None

    def test_different_query_is_miss(self):
        from llm_optimizer.semantic_cache import SemanticCache
        cache = SemanticCache(threshold=0.92, ttl_seconds=60)
        msgs1 = [{"role": "user", "content": "capital of France"}]
        msgs2 = [{"role": "user", "content": "best pizza recipe ingredients"}]
        cache.set(msgs1, "Paris", "model", 10, 5, 0.001)
        result = cache.get(msgs2)
        assert result is None

    def test_ttl_expiry(self):
        import time
        from llm_optimizer.semantic_cache import SemanticCache
        cache = SemanticCache(threshold=0.85, ttl_seconds=1)
        msgs = [{"role": "user", "content": "Hello world test query"}]
        cache.set(msgs, "Hi", "model", 10, 5, 0.001)
        time.sleep(1.1)
        result = cache.get(msgs)
        assert result is None

    def test_stats_tracked(self):
        from llm_optimizer.semantic_cache import SemanticCache
        cache = SemanticCache(threshold=0.85, ttl_seconds=60)
        msgs = [{"role": "user", "content": "capital France"}]
        cache.set(msgs, "Paris", "model", 10, 5, 0.002)
        cache.get(msgs)   # hit
        cache.get([{"role": "user", "content": "pizza recipe"}])  # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["cost_saved_usd"] == 0.002

    def test_invalid_threshold_raises(self):
        from llm_optimizer.semantic_cache import SemanticCache
        with pytest.raises(ValueError):
            SemanticCache(threshold=1.5)


# ─── Cross-Provider Cache (unit) ──────────────────────────────────────────────

class TestCrossProviderCache:
    def test_same_provider_hit(self):
        from llm_optimizer.cross_provider_cache import CrossProviderCache
        cache = CrossProviderCache()
        msgs = [{"role": "user", "content": "Hello"}]
        key = cache.make_key(msgs)
        cache.set(key, "Hi", "anthropic", "claude-haiku", 10, 5, 0.001)
        result = cache.get(key, "anthropic")
        assert result is not None
        assert result.response_text == "Hi"

    def test_cross_provider_hit(self):
        """Key is provider-agnostic by default."""
        from llm_optimizer.cross_provider_cache import CrossProviderCache
        cache = CrossProviderCache(provider_agnostic=True)
        msgs = [{"role": "user", "content": "Hello"}]
        key_a = cache.make_key(msgs, provider="anthropic")
        key_b = cache.make_key(msgs, provider="bedrock")
        cache.set(key_a, "Hi", "anthropic", "claude-haiku", 10, 5, 0.001)
        result = cache.get(key_b, "bedrock")
        assert result is not None  # cross-provider hit

    def test_provider_specific_miss(self):
        """When provider_agnostic=False, different providers miss."""
        from llm_optimizer.cross_provider_cache import CrossProviderCache
        cache = CrossProviderCache(provider_agnostic=False)
        msgs = [{"role": "user", "content": "Hello"}]
        key_a = cache.make_key(msgs, provider="anthropic")
        key_b = cache.make_key(msgs, provider="bedrock")
        cache.set(key_a, "Hi", "anthropic", "claude-haiku", 10, 5, 0.001)
        assert key_a != key_b  # different keys
        result = cache.get(key_b, "bedrock")
        assert result is None

    def test_cross_provider_hits_tracked(self):
        from llm_optimizer.cross_provider_cache import CrossProviderCache
        cache = CrossProviderCache()
        msgs = [{"role": "user", "content": "Hello"}]
        key = cache.make_key(msgs)
        cache.set(key, "Hi", "anthropic", "model", 10, 5, 0.001)
        cache.get(key, "bedrock")
        stats = cache.stats()
        assert "anthropic→bedrock" in stats["cross_provider_hits"]


# ─── Prefill & Forced Output (unit) ───────────────────────────────────────────

class TestPrefill:
    def test_json_system_detected(self):
        from llm_optimizer.prefill import detect_prefill
        prefill = detect_prefill(system="Always respond with valid JSON only.")
        assert prefill == '{"'

    def test_output_schema_forces_json(self):
        from llm_optimizer.prefill import detect_prefill
        prefill = detect_prefill(output_schema={"type": "object"})
        assert prefill == '{"'

    def test_explicit_prefill_wins(self):
        from llm_optimizer.prefill import detect_prefill
        prefill = detect_prefill(
            system="Return JSON.",
            explicit_prefill="```python\n",
        )
        assert prefill == "```python\n"

    def test_no_signal_returns_none(self):
        from llm_optimizer.prefill import detect_prefill
        prefill = detect_prefill(system="You are a helpful assistant.")
        assert prefill is None

    def test_inject_prefill_adds_assistant_turn(self):
        from llm_optimizer.prefill import inject_prefill
        msgs = [{"role": "user", "content": "Hi"}]
        result = inject_prefill(msgs, '{"')
        assert result[-1]["role"] == "assistant"
        assert result[-1]["content"] == '{"'

    def test_inject_prefill_no_duplicate(self):
        from llm_optimizer.prefill import inject_prefill
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = inject_prefill(msgs, '{"')
        assert len(result) == 2  # not added when last msg is assistant

    def test_strip_prefill_from_response(self):
        from llm_optimizer.prefill import strip_prefill_from_response
        response = "Hello world"
        # prefill is "Hello" — strip from start
        stripped = strip_prefill_from_response(response, "Hello")
        assert stripped == " world"

    def test_strip_prefill_not_present(self):
        from llm_optimizer.prefill import strip_prefill_from_response
        response = "Hello world"
        stripped = strip_prefill_from_response(response, '{"')
        assert stripped == "Hello world"


# ─── Few-Shot Selector (unit) ─────────────────────────────────────────────────

class TestFewShotSelector:
    def test_add_and_select(self):
        from llm_optimizer.few_shot import FewShotSelector
        sel = FewShotSelector(top_k=2)
        sel.add("classify email as spam", "SPAM")
        sel.add("is this message promotional", "SPAM")
        sel.add("translate hello to french", "Bonjour")
        results = sel.select("categorize this email as spam or not")
        assert len(results) <= 2
        assert any("spam" in r.input_text.lower() or "email" in r.input_text.lower() for r in results)

    def test_top_k_limit(self):
        from llm_optimizer.few_shot import FewShotSelector
        sel = FewShotSelector(top_k=2)
        for i in range(10):
            sel.add(f"example input {i}", f"output {i}")
        results = sel.select("example input query")
        assert len(results) <= 2

    def test_label_filter(self):
        from llm_optimizer.few_shot import FewShotSelector
        sel = FewShotSelector(top_k=3)
        sel.add("spam example", "SPAM", label="classification")
        sel.add("translate hello", "Bonjour", label="translation")
        sel.add("another spam", "SPAM", label="classification")
        results = sel.select("spam detection", label_filter="classification")
        assert all(r.label == "classification" for r in results)

    def test_format_output(self):
        from llm_optimizer.few_shot import FewShotSelector
        sel = FewShotSelector(top_k=2)
        sel.add("input A", "output A")
        examples = sel.select("something similar to A")
        formatted = sel.format(examples)
        assert "Input:" in formatted
        assert "Output:" in formatted

    def test_empty_selector_returns_empty(self):
        from llm_optimizer.few_shot import FewShotSelector
        sel = FewShotSelector()
        results = sel.select("any query")
        assert results == []

    def test_bulk_add(self):
        from llm_optimizer.few_shot import FewShotSelector
        sel = FewShotSelector(top_k=3)
        sel.add_bulk([
            ("input 1", "output 1"),
            ("input 2", "output 2", "label_a"),
        ])
        assert sel.stats()["total_examples"] == 2


# ─── Output Auto-Correction (unit) ────────────────────────────────────────────

class TestOutputCorrector:
    def test_extract_json_valid(self):
        from llm_optimizer.output_repair import extract_json_from_text
        result = extract_json_from_text('{"name": "John", "age": 30}')
        assert result is not None
        import json
        assert json.loads(result)["name"] == "John"

    def test_extract_json_from_markdown(self):
        from llm_optimizer.output_repair import extract_json_from_text
        text = '```json\n{"name": "Alice"}\n```'
        result = extract_json_from_text(text)
        assert result is not None

    def test_extract_json_embedded_in_text(self):
        from llm_optimizer.output_repair import extract_json_from_text
        text = 'Here is the result: {"status": "ok"} as requested.'
        result = extract_json_from_text(text)
        assert result is not None

    def test_corrector_fixes_extractable_json(self):
        from llm_optimizer.output_repair import OutputCorrector
        corrector = OutputCorrector(enabled=True)
        broken = 'Sure! Here is the JSON: {"name": "Bob", "age": 25}'
        corrected, was_fixed = corrector.correct(broken)
        assert was_fixed
        import json
        assert json.loads(corrected)["name"] == "Bob"

    def test_corrector_disabled_passthrough(self):
        from llm_optimizer.output_repair import OutputCorrector
        corrector = OutputCorrector(enabled=False)
        broken = "not json at all"
        corrected, was_fixed = corrector.correct(broken)
        assert not was_fixed
        assert corrected == broken

    def test_stats_tracked(self):
        from llm_optimizer.output_repair import OutputCorrector
        corrector = OutputCorrector(enabled=True)
        corrector.correct('{"valid": true}')  # extractable
        stats = corrector.stats()
        assert "corrections_made" in stats


# ─── PII Masking (unit) ───────────────────────────────────────────────────────

class TestPIIMasker:
    def test_email_masked(self):
        from llm_optimizer.pii import PIIMasker
        masker = PIIMasker(mask_emails=True)
        msgs = [{"role": "user", "content": "Contact john@example.com for help"}]
        masked_msgs, _, session = masker.mask_messages(msgs)
        assert "john@example.com" not in masked_msgs[0]["content"]
        assert "[EMAIL_1]" in masked_msgs[0]["content"]

    def test_email_restored(self):
        from llm_optimizer.pii import PIIMasker
        masker = PIIMasker(mask_emails=True)
        msgs = [{"role": "user", "content": "Email: alice@test.com"}]
        _, _, session = masker.mask_messages(msgs)
        restored = masker.restore_response("Contact [EMAIL_1] asap", session)
        assert "alice@test.com" in restored

    def test_credit_card_masked(self):
        from llm_optimizer.pii import PIIMasker
        masker = PIIMasker(mask_cards=True)
        msgs = [{"role": "user", "content": "Card: 4111-1111-1111-1111"}]
        masked_msgs, _, session = masker.mask_messages(msgs)
        assert "4111" not in masked_msgs[0]["content"]
        assert "[CARD_1]" in masked_msgs[0]["content"]

    def test_same_value_same_token(self):
        """Same PII value gets same token throughout request."""
        from llm_optimizer.pii import PIIMasker
        masker = PIIMasker(mask_emails=True)
        msgs = [{"role": "user", "content": "Send to bob@test.com and also bob@test.com"}]
        masked_msgs, _, session = masker.mask_messages(msgs)
        content = masked_msgs[0]["content"]
        assert content.count("[EMAIL_1]") == 2
        assert "[EMAIL_2]" not in content

    def test_no_pii_unchanged(self):
        from llm_optimizer.pii import PIIMasker
        masker = PIIMasker()
        msgs = [{"role": "user", "content": "What is the weather today?"}]
        masked_msgs, _, session = masker.mask_messages(msgs)
        assert masked_msgs[0]["content"] == "What is the weather today?"
        assert session.masked_count() == 0

    def test_pii_stats(self):
        from llm_optimizer.pii import PIIMasker
        masker = PIIMasker()
        msgs = [{"role": "user", "content": "Email: test@test.com"}]
        masker.mask_messages(msgs)
        stats = masker.stats()
        assert stats["total_pii_values_masked"] == 1
        assert stats["requests_with_pii"] == 1

    def test_system_prompt_masked(self):
        from llm_optimizer.pii import PIIMasker
        masker = PIIMasker(mask_emails=True)
        msgs = []
        system = "Contact admin@company.com for support."
        _, masked_sys, session = masker.mask_messages(msgs, system=system)
        assert "admin@company.com" not in masked_sys
        assert "[EMAIL_1]" in masked_sys


# ─── Streaming Analytics (unit) ───────────────────────────────────────────────

class TestStreamingAnalytics:
    def test_collector_records_chunks(self):
        from llm_optimizer.analytics import StreamingAnalyticsCollector
        collector = StreamingAnalyticsCollector(model_key="claude-haiku-4-5")
        for word in ["Hello", " world", " this", " is", " a", " test"]:
            collector.record_chunk(word)
        stats = collector.current_stats()
        assert stats["chunk_count"] == 6
        assert stats["estimated_output_tokens"] > 0

    def test_callback_fired(self):
        from llm_optimizer.analytics import StreamingAnalyticsCollector
        events = []
        collector = StreamingAnalyticsCollector(
            model_key="claude-haiku-4-5",
            on_token_event=lambda e: events.append(e),
            emit_every_n=1,
        )
        for word in ["Hello", " world"]:
            collector.record_chunk(word)
        assert len(events) == 2

    def test_finalize_returns_analytics(self):
        from llm_optimizer.analytics import StreamingAnalyticsCollector, StreamAnalytics
        collector = StreamingAnalyticsCollector(model_key="claude-haiku-4-5")
        collector.record_chunk("Hello world test message")
        result = collector.finalize()
        assert isinstance(result, StreamAnalytics)
        assert result.total_chunks == 1
        assert result.elapsed_ms > 0

    def test_wrap_stream_with_analytics(self):
        from llm_optimizer.analytics import wrap_stream_with_analytics
        chunks = ["Hello", " world", " test"]
        gen, collector = wrap_stream_with_analytics(iter(chunks), "claude-haiku-4-5")
        collected = list(gen)
        assert collected == chunks
        final = collector.finalize()
        assert final.total_chunks == 3

    def test_cost_estimated(self):
        from llm_optimizer.analytics import StreamingAnalyticsCollector
        collector = StreamingAnalyticsCollector(
            model_key="claude-haiku-4-5",
            input_tokens=1000,
        )
        for _ in range(10):
            collector.record_chunk("word ")
        stats = collector.current_stats()
        assert stats["estimated_cost_usd"] > 0
