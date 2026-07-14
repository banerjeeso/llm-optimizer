# llm-optimizer

**Reduce LLM API costs by up to 90%** — a Python library that wraps Anthropic, OpenAI, and Google clients and transparently applies six cost-saving techniques.

```python
import anthropic
from llm_optimizer import OptimizedClient

client = OptimizedClient(anthropic_client=anthropic.Anthropic())

response = client.complete(
    messages=[{"role": "user", "content": "Summarize this article..."}],
    system="You are a helpful assistant.",
)

client.print_summary()
# 💰 Total cost: $0.000842  |  Total saved: $0.007580  (90% savings)
```

![tests](https://img.shields.io/badge/tests-62%20passing-brightgreen) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green)

> **v0.2.0** — streaming support, LLM-based routing, TF-IDF document compression,
> externalized pricing, retry/fallback logic. See [CHANGELOG](#changelog).

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Six Optimization Techniques](#six-optimization-techniques)
- [Streaming](#streaming)
- [Error Handling and Fallback Behavior](#error-handling-and-fallback-behavior)
- [Limitations](#limitations)
- [Configuration Reference](#configuration-reference)
- [Pricing Data](#pricing-data)
- [Available Models](#available-models)
- [Test Suite](#test-suite)
- [Changelog](#changelog)

---

## Installation

```bash
# Anthropic only (minimum)
pip install llm-optimizer

# With exact token counting (recommended for production)
pip install llm-optimizer[tiktoken]

# All providers + exact token counting
pip install llm-optimizer[all]
```

---

## Quick Start

```python
import anthropic
from llm_optimizer import OptimizedClient

client = OptimizedClient(
    anthropic_client=anthropic.Anthropic(),
    enable_caching=True,        # default: True
    enable_routing=True,        # default: True
    enable_optimization=True,   # default: True
    enable_compression=False,   # default: False — read warning below before enabling
    enable_tracking=True,       # default: True
)

response = client.complete(
    messages=[{"role": "user", "content": "Is this email spam? 'You won a prize!'"}],
    system="You are a spam classifier. Reply with SPAM or NOT SPAM.",
)
print(response.content[0].text)

client.print_summary()
```

---

## Six Optimization Techniques

### 1. 💾 Prompt Caching — up to 90% off

Automatically injects `cache_control` breakpoints at optimal positions in system prompts,
few-shot examples, and long conversation history. Anthropic charges 10% of the normal
input price for cached tokens.

```python
# Build an optimally structured, cached system prompt
system = client.build_cached_system(
    base_instructions="You are an expert legal analyst...",  # cached — charged once
    few_shot_examples="Example 1: ...\nExample 2: ...",      # cached separately
    knowledge_base="[50K token reference document]",         # cached separately
)

# First call: pays full price for system prompt
response1 = client.complete(messages=[...], system=system)

# Second+ calls: system served from cache at 10% of cost
response2 = client.complete(messages=[...], system=system)  # 90% cheaper
```

**How it works:** The library injects `cache_control: {type: "ephemeral"}` on content
blocks that exceed 1,024 tokens, respecting Anthropic's 4-breakpoint limit. For OpenAI,
prefix caching is automatic on inputs ≥ 1,024 tokens — no markup needed.

---

### 2. 🔀 Model Routing — 60–80% off

Routes each request to the cheapest model capable of handling its complexity.

**Two modes:**

**Heuristic routing (default)** — free, fast, keyword + token-count based:
```python
# Routed to Haiku automatically (simple task)
client.complete(messages=[{"role": "user", "content": "Classify this as spam or not"}])
```

**LLM classifier routing (recommended for production)** — uses Haiku to assess
complexity before routing. Costs ~15 tokens per call but correctly handles
intent-heavy short prompts that heuristics misclassify:

```python
client = OptimizedClient(
    anthropic_client=anthropic.Anthropic(),
    enable_llm_classifier=True,  # ~$0.000015 per routing call; worth it
)

# "Explain the constitutional implications of this clause" — 8 words, but EXPERT
# Heuristic: misrouted to SIMPLE. LLM classifier: correctly routed to EXPERT.
```

**Audit the routing decision:**
```python
from llm_optimizer import ModelRouter
router = ModelRouter(preferred_provider=Provider.ANTHROPIC)
print(router.explain("Explain the constitutional implications of this clause"))
# {
#   "detected_complexity": "expert",
#   "routed_model": "claude-opus-4",
#   "keyword_signals_fired": {"expert": ["constitutional", "implications"]},
#   "token_count": 9,
#   "warning": "Heuristics cannot detect intent-heavy short prompts. Use enable_llm_classifier=True..."
# }
```

**Complexity override (escape hatch):**
```python
# Use only when you know the router is wrong for your specific case.
# When you specify complexity manually, you take responsibility for correctness.
from llm_optimizer import TaskComplexity
client.complete(messages=[...], complexity=TaskComplexity.EXPERT)
```

---

### 3. ✂️ Prompt Optimization — 5–20% off

Strips filler phrases, normalizes whitespace, and replaces verbose constructs.
Code blocks are always preserved. Conversation history is trimmed beyond N turns.

```python
from llm_optimizer import PromptOptimizer

opt = PromptOptimizer()
result = opt.optimize(
    "In order to complete this task, please note that you should carefully "
    "analyze the following.   It is important to note that accuracy matters."
)
# → "To complete this task, carefully analyze the following. Accuracy matters."
# Saved: 12 tokens (24%)
```

What gets removed: filler phrases (`"please note that"`, `"in order to"`, `"due to the fact that"`),
redundant whitespace, verbose hedges. What is never touched: code blocks, factual content,
user-specified phrasing inside quotes.

---

### 4. 📦 Batch Processing — 50% off

Queue non-urgent requests and submit via Anthropic's Message Batch API (50% cheaper,
results within minutes to hours).

```python
client = OptimizedClient(
    anthropic_client=anthropic.Anthropic(),
    enable_batching=True,
)

# Queue throughout the day
for item in items:
    client.queue(
        custom_id=item["id"],
        messages=[{"role": "user", "content": f"Summarize: {item['text']}"}],
        max_tokens=200,
    )

# Submit as one batch — 50% cheaper than real-time
batch_id = client.submit_batch()

# Poll when convenient
results = client.poll_batch(batch_id, wait=True, timeout_seconds=3600)
for r in results:
    print(f"{r.custom_id}: {'✓' if r.success else '✗'} {r.content or r.error}")
```

---

### 5. 📄 Document Compression — variable savings

> ⚠️ **Accuracy warning — read before enabling.**
>
> Compression removes content. There is **no guarantee** removed content is irrelevant
> to your query. Answers may be incomplete or incorrect. This feature trades accuracy
> for cost. Enable only after validating it works for your document type.
>
> Do **not** use for: legal/compliance documents, code files, or any task where
> completeness is required for correctness.

```python
client = OptimizedClient(
    anthropic_client=anthropic.Anthropic(),
    enable_compression=True,        # opt-in; emits a UserWarning at init
    max_document_tokens=4000,
    compression_strategy="extractive",  # best accuracy; requires query
)

# Pass documents — compressed before sending
response = client.complete(
    messages=[{"role": "user", "content": "What are the payment terms?"}],
    documents=[very_long_contract],  # compressed to 4K tokens
)
```

**Three strategies, ranked by accuracy:**

| Strategy | Accuracy | Requires query | How it works |
|----------|----------|----------------|--------------|
| `extractive` | Best | Yes | TF-IDF paragraph scoring against query |
| `smart` | Medium | No | Keeps first 60% + last 20% by position |
| `truncate` | Lowest | No | Hard cutoff at token limit |

All strategies append a visible `[⚠️ COMPRESSION WARNING]` to compressed output
so downstream code can detect compression occurred.

---

### 6. 📊 Cost Tracking and Analytics

```python
client = OptimizedClient(
    anthropic_client=anthropic.Anthropic(),
    persist_tracking="usage.jsonl",  # survives restarts
)

# ... run your application ...

summary = client.summary()
print(f"Total cost:    ${summary['total_cost_usd']:.4f}")
print(f"Total saved:   ${summary['total_cost_saved_usd']:.4f} ({summary['savings_pct']}%)")
print(f"By model:      {summary['by_model']}")
print(f"Optimizations: {summary['optimizations_applied']}")

client.print_summary()  # formatted console output
```

---

## Streaming

Use `client.stream()` for user-facing applications. `client.complete()` blocks
until the full response arrives and should not be used where latency matters.

```python
# Basic streaming
with client.stream(
    messages=[{"role": "user", "content": "Write a poem about clouds"}],
    system="You are a poet.",
    max_tokens=512,
) as stream:
    for text_chunk in stream:
        print(text_chunk, end="", flush=True)

print()  # newline after stream ends

# Access token usage after stream is fully consumed
usage = stream.usage()
if usage:
    print(f"Tokens used: {usage['input_tokens']} in, {usage['output_tokens']} out")
```

All optimizations (caching, routing, prompt optimization) apply identically
to `stream()` and `complete()`.

**Provider support:**
- ✅ Anthropic (Claude)
- ✅ OpenAI (GPT)
- ❌ Google (Gemini) — not yet implemented; raises `NotImplementedError`

---

## Error Handling and Fallback Behavior

The library handles the most common production failure modes automatically.

| Failure | Behavior |
|---------|----------|
| Rate limit (429) | Retries with exponential backoff, up to `max_retries` (default: 3) |
| Model overloaded (529) | Falls back to next cheapest capable model; emits `UserWarning` |
| Exhausted retries | Raises the original provider exception |
| Non-retriable error | Raises immediately (no retry waste) |
| LLM classifier failure | Silently falls back to heuristic routing |
| Cache miss | Transparent — Anthropic handles server-side |
| Batch API error | Re-queues requests; raises `RuntimeError` after `max_retries` |

```python
client = OptimizedClient(
    anthropic_client=anthropic.Anthropic(),
    max_retries=3,          # default
    retry_base_delay=1.0,   # seconds; doubles on each retry (1s, 2s, 4s)
)
```

To disable retries entirely:
```python
client = OptimizedClient(..., max_retries=0)
```

---

## Limitations

**Known limitations — read before adopting in production.**

**Model router — heuristic mode:**
Keyword-based heuristics misclassify intent-heavy short prompts. A prompt like
`"Explain the constitutional implications of this clause"` is 8 words but requires
expert-level reasoning. The heuristic will route it to a cheap model.
Use `enable_llm_classifier=True` to avoid this class of error.

**Document compression:**
All compression strategies are lossy. There is no semantic guarantee about what
is preserved. The `extractive` strategy is the most accurate but still removes
content. Do not use for documents where every sentence may be relevant.

**Streaming — no tracking after stream:**
`client.stream()` records estimated token counts at call time. Exact counts
are available via `stream.usage()` only after the stream is fully consumed,
and are not automatically written to the cost tracker.

**Token counting without tiktoken:**
The default heuristic estimator is within ~20-30% of actual token counts for
English prose. Install `tiktoken` for exact counts: `pip install llm-optimizer[tiktoken]`.

**Pricing staleness:**
Prices are read from `pricing.json`. The library warns if data is older than
30 days. Always verify estimates against provider pricing pages before making
financial decisions.

**OpenAI and Google — partially tested:**
Provider integrations are implemented but not tested end-to-end in CI.
Anthropic is the primary tested provider.

**No async support yet:**
`client.complete()` and `client.stream()` are synchronous. Async support
(`acomplete()`, `astream()`) is planned for v0.3.0.

---

## Configuration Reference

```python
client = OptimizedClient(
    # Provider clients — pass whichever you use
    anthropic_client=anthropic.Anthropic(),
    openai_client=openai.OpenAI(),           # optional
    google_client=genai.GenerativeModel(),   # optional

    # Feature flags
    enable_caching=True,          # prompt cache_control injection
    enable_routing=True,          # auto model selection
    enable_optimization=True,     # prompt filler stripping
    enable_compression=False,     # ⚠️ lossy — opt-in only
    enable_tracking=True,         # cost analytics
    enable_batching=False,        # batch API — opt-in

    # Routing
    preferred_provider=Provider.ANTHROPIC,
    default_model=None,                 # None = auto-route
    enable_llm_classifier=False,        # accurate routing; costs ~15 tokens/call

    # Prompt optimization
    aggressive_optimization=False,      # more aggressive phrase removal
    max_history_turns=10,               # trim conversation beyond this

    # Compression (only used if enable_compression=True)
    max_document_tokens=4000,
    compression_strategy="smart",       # "extractive" | "smart" | "truncate"

    # Cost tracking
    persist_tracking="usage.jsonl",     # optional JSONL persistence

    # Reliability
    max_retries=3,                      # 0 to disable
    retry_base_delay=1.0,               # seconds; doubles per retry

    # Batching
    batch_default_model=None,           # defaults to claude-haiku
)
```

---

## Pricing Data

Prices are stored in `llm_optimizer/pricing.json` — not hardcoded.

```json
{
  "_meta": {
    "version": "2026-04-03",
    "last_updated": "2026-04-03",
    "warning": "Provider pricing changes without notice...",
    "sources": {
      "anthropic": "https://www.anthropic.com/pricing",
      "openai": "https://openai.com/pricing",
      "google": "https://ai.google.dev/pricing"
    }
  }
}
```

The library emits a `UserWarning` automatically if pricing data is older than 30 days.
Update pricing by editing `pricing.json` and bumping `last_updated`.

```python
from llm_optimizer.models import pricing_info
print(pricing_info())
# {"version": "2026-04-03", "last_updated": "2026-04-03", "sources": {...}}
```

---

## Available Models

> ⚠️ Prices shown as of `pricing.json` version `2026-04-03`. Verify before relying on estimates.

| Key | Model | Provider | Input/1M | Cached/1M | Output/1M | Complexity |
|-----|-------|----------|----------|-----------|-----------|------------|
| `claude-haiku-4-5` | Claude Haiku 4.5 | Anthropic | $1.00 | $0.10 | $5.00 | Simple, Medium |
| `claude-haiku-3-5` | Claude Haiku 3.5 | Anthropic | $0.80 | $0.08 | $4.00 | Simple, Medium |
| `claude-sonnet-4` | Claude Sonnet 4.5 | Anthropic | $3.00 | $0.30 | $15.00 | Simple–Complex |
| `claude-opus-4` | Claude Opus 4.5 | Anthropic | $5.00 | $0.50 | $25.00 | All |
| `gpt-5-mini` | GPT-5 Mini | OpenAI | $0.25 | $0.025 | $2.00 | Simple, Medium |
| `gpt-5` | GPT-5 | OpenAI | $2.50 | $0.25 | $15.00 | All |
| `gemini-flash` | Gemini 2.5 Flash | Google | $0.30 | $0.03 | $2.50 | Simple, Medium |
| `gemini-pro` | Gemini 2.5 Pro | Google | $1.25 | $0.125 | $10.00 | All |

---

## Test Suite

```bash
# Unit tests only (no API keys needed)
pip install pytest
python -m pytest tests/ -v -m "not integration"

# Integration tests (requires ANTHROPIC_API_KEY)
python -m pytest tests/ -v -m integration
```

**What the unit tests cover (62 tests, no real API calls):**
- Token estimation logic
- Model pricing calculations
- Heuristic routing decisions and regression cases
- LLM classifier routing (mocked Anthropic client)
- Prompt optimization transforms
- Document compression — all three strategies including TF-IDF scoring
- Cache_control block injection
- Cost tracker arithmetic, persistence, and thread safety
- Retry and fallback logic (mocked exceptions)
- Streaming module construction

**What unit tests do NOT cover (requires real API):**
- Actual cache hits and cache write tokens from Anthropic
- LLM classifier calls against live API
- Batch submission, polling, and result collection
- End-to-end OpenAI and Google provider calls
- Streaming token delivery timing

---

## Project Structure

```
llm_optimizer/
├── __init__.py         # Public API surface
├── client.py           # OptimizedClient — main entry point
├── streaming.py        # StreamResult, streaming request builders
├── models.py           # Model registry — loads from pricing.json
├── pricing.json        # Externalized pricing data with version stamp
├── router.py           # Heuristic + LLM-classifier model routing
├── optimizer.py        # Prompt optimization + TF-IDF document compression
├── caching.py          # cache_control injection
├── batcher.py          # Anthropic Batch API wrapper
└── tracker.py          # Cost tracking + JSONL persistence

examples/
├── 01_basic_usage.py
├── 02_document_analysis.py
└── 03_batch_processing.py

tests/
└── test_llm_optimizer.py   # 62 unit tests, all mocked
```

---

## Changelog

## v0.3.1

### Added
- AWS Bedrock support via `BedrockClient` and `make_bedrock_client` factory
- Both auth patterns supported — AWS profile (`profile_name="ai-core"`) and env vars / IAM role
- Bedrock model registry — Claude Haiku 4.5 and Sonnet 4.5 with correct AWS model IDs
- Bedrock pricing in `pricing.json` — AWS list rates, separate from Anthropic.com pricing
- `discount_pct` parameter on `BedrockClient` for negotiated AWS account discounts
- `Provider.BEDROCK` added to provider enum
- Bedrock entries added to `TIER_DEFAULTS` for all four complexity tiers
- 17 new unit tests covering Bedrock integration

### Fixed
- GitHub repository URL corrected in `pyproject.toml`
- `cache_control` blocks now stripped automatically before Bedrock requests

### Notes
- Prompt caching is not supported on AWS Bedrock — `cache_control` is silently removed
- Bedrock batch processing uses Bedrock Batch Inference, not Anthropic Batch API (planned for v0.4.0)

---

## v0.3.0

### Added
- AWS Bedrock scaffolding (incomplete — superseded by v0.3.1)

### v0.2.0

- **Streaming** — `client.stream()` added for Anthropic and OpenAI
- **LLM classifier routing** — opt-in via `enable_llm_classifier=True`; fixes misclassification of short intent-heavy prompts
- **Router audit trail** — `router.explain()` returns full decision trace with warning when in heuristic mode
- **TF-IDF document compression** — new `extractive` strategy scores paragraphs by relevance to query
- **Compression accuracy warning** — `enable_compression` defaults to `False`; emits `UserWarning` on init; all strategies append visible warning to compressed text
- **Externalized pricing** — prices moved to `pricing.json` with `last_updated` stamp; auto-warns when stale
- **Retry and fallback** — exponential backoff on rate limits; automatic model fallback on overload
- **Test suite expanded** — 62 tests (up from 36); clearly documents what is and isn't mocked

### v0.1.0

- Initial release — prompt caching, heuristic routing, prompt optimization, batch processing, document compression, cost tracking

---

## License

MIT
