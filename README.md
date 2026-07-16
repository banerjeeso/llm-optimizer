# llm-optimizer

**Reduce LLM API costs by up to 90%** — a Python library that wraps Anthropic, OpenAI, and AWS Bedrock clients and transparently applies cost-saving techniques automatically.

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

![tests](https://img.shields.io/badge/tests-172%20passing-brightgreen) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green)

> **v0.4.0** — budget enforcement, output cache, semantic cache, Bedrock cachePoint translation,
> few-shot selector, JSON auto-correction, streaming analytics. See [CHANGELOG](#changelog).

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Optimization Techniques](#optimization-techniques)
- [Streaming](#streaming)
- [Error Handling and Fallback Behavior](#error-handling-and-fallback-behavior)
- [Limitations](#limitations)
- [Configuration Reference](#configuration-reference)
- [Pricing Data](#pricing-data)
- [Available Models](#available-models)
- [Test Suite](#test-suite)
- [Project Structure](#project-structure)
- [Changelog](#changelog)

---

## Installation

```bash
# Anthropic only (minimum)
pip install llm-optimizer

# With AWS Bedrock support
pip install llm-optimizer[bedrock]

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
    enable_compression=False,   # default: False — read warning before enabling
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

## Optimization Techniques

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
    enable_llm_classifier=True,  # ~$0.000015 per routing call
)
# "Explain the constitutional implications of this clause" — 8 words, but EXPERT
# Heuristic: misrouted to SIMPLE. LLM classifier: correctly routed to EXPERT.
```

**Audit the routing decision:**
```python
from llm_optimizer import ModelRouter, Provider

router = ModelRouter(preferred_provider=Provider.ANTHROPIC)
print(router.explain("Explain the constitutional implications of this clause"))
# {
#   "detected_complexity": "expert",
#   "routed_model": "claude-opus-4",
#   "keyword_signals_fired": {"expert": ["constitutional", "implications"]},
#   "token_count": 9,
#   "warning": "Heuristics cannot detect intent-heavy short prompts..."
# }
```

**Complexity override (escape hatch):**
```python
# Use only when you know the router is wrong for your specific case
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

---

### 4. 📦 Batch Processing — 50% off

Queue non-urgent requests and submit via Anthropic's Message Batch API (50% cheaper,
results within minutes to hours).

```python
client = OptimizedClient(
    anthropic_client=anthropic.Anthropic(),
    enable_batching=True,
)

for item in items:
    client.queue(
        custom_id=item["id"],
        messages=[{"role": "user", "content": f"Summarize: {item['text']}"}],
        max_tokens=200,
    )

batch_id = client.submit_batch()
results = client.poll_batch(batch_id, wait=True, timeout_seconds=3600)
for r in results:
    print(f"{r.custom_id}: {'✓' if r.success else '✗'} {r.content or r.error}")
```

---

### 5. 📄 Document Compression — variable savings

> ⚠️ **Accuracy warning — read before enabling.**
>
> Compression removes content. There is **no guarantee** removed content is irrelevant
> to your query. Answers may be incomplete or incorrect. Enable only after validating
> it works for your document type. Do **not** use for legal/compliance documents,
> code files, or any task where completeness is required.

```python
client = OptimizedClient(
    anthropic_client=anthropic.Anthropic(),
    enable_compression=True,        # opt-in; emits a UserWarning at init
    max_document_tokens=4000,
    compression_strategy="extractive",
)

response = client.complete(
    messages=[{"role": "user", "content": "What are the payment terms?"}],
    documents=[very_long_contract],
)
```

| Strategy | Accuracy | Requires query | How it works |
|---|---|---|---|
| `extractive` | Best | Yes | TF-IDF paragraph scoring against query |
| `smart` | Medium | No | Keeps first 60% + last 20% by position |
| `truncate` | Lowest | No | Hard cutoff at token limit |

All strategies append a visible `[⚠️ COMPRESSION WARNING]` to compressed output.

---

### 6. 📊 Cost Tracking and Analytics

```python
client = OptimizedClient(
    anthropic_client=anthropic.Anthropic(),
    persist_tracking="usage.jsonl",
)

summary = client.summary()
print(f"Total cost:  ${summary['total_cost_usd']:.4f}")
print(f"Total saved: ${summary['total_cost_saved_usd']:.4f} ({summary['savings_pct']}%)")

client.print_summary()  # formatted console output
```

---

### 7. 💰 Budget Enforcement

Hard cost limits enforced before any API call is made. The request is never sent
if it would exceed the configured limit — zero cost on blocked requests.

```python
from llm_optimizer.budget import BudgetExceededError

client = OptimizedClient(
    anthropic_client=anthropic.Anthropic(),
    max_cost_per_request=0.01,   # block if single request > $0.01
    monthly_budget=50.00,        # block when month total > $50
    on_budget_warning=lambda pct, t: print(f"⚠️ {pct:.0f}% of budget used"),
)

try:
    response = client.complete(messages=[...])
except BudgetExceededError as e:
    print(f"Blocked: {e}")

print(client.budget_status())
# {"total_spent_usd": 12.40, "monthly_remaining": 37.60, "monthly_used_pct": 24.8}
```

---

### 8. 🗄️ Output Cache — 100% savings on repeated prompts

Returns cached API responses for identical prompts at zero cost.

```python
client = OptimizedClient(
    anthropic_client=anthropic.Anthropic(),
    enable_output_cache=True,
    output_cache_backend="sqlite",  # or "memory"
    output_cache_ttl=3600,
    output_cache_max_size=1000,
)

# First call — hits API
r1 = client.complete(messages=[{"role": "user", "content": "What is 2+2?"}])

# Second call — served from cache, zero cost
r2 = client.complete(messages=[{"role": "user", "content": "What is 2+2?"}])

print(client.cache_stats())
# {"hits": 1, "misses": 1, "hit_rate_pct": 50.0, "cost_saved_usd": 0.000012}
```

| Backend | Persistence | Best for |
|---|---|---|
| `memory` | Lost on restart | Dev/test, short sessions |
| `sqlite` | Survives restarts | Production, shared across processes |

---

### 9. 🔍 Semantic (Fuzzy) Cache — 100% savings on paraphrases

Unlike the output cache which requires identical prompts, the semantic cache uses
TF-IDF cosine similarity to match prompts that mean the same thing. No external
dependencies required.

```python
client = OptimizedClient(
    anthropic_client=anthropic.Anthropic(),
    enable_semantic_cache=True,
    semantic_cache_threshold=0.92,  # 0-1; higher = stricter matching
    semantic_cache_ttl=3600,
)

r1 = client.complete(messages=[{"role": "user", "content": "What is the capital of France?"}])
r2 = client.complete(messages=[{"role": "user", "content": "France's capital city?"}])
# r2 served from cache — zero cost

print(client.semantic_cache_stats())
# {"hits": 1, "misses": 1, "hit_rate_pct": 50.0, "cost_saved_usd": 0.000012}
```

**Threshold guide:** `0.95` = near-identical only. `0.85` = loose paraphrases. Start at `0.92`.

---

### 10. 🔀 Cross-Provider Cache

Cache keys are derived from prompt content only — not the provider or model. A response
cached on Anthropic is returned when the same prompt is sent through Bedrock.

```python
client = OptimizedClient(
    anthropic_client=anthropic.Anthropic(),
    enable_cross_provider_cache=True,
)

# Cached on Anthropic
r1 = client.complete(messages=[...], force_provider=Provider.ANTHROPIC)

# Hit served from cache even though provider changed
r2 = client.complete(messages=[...], force_provider=Provider.BEDROCK)

print(client.cross_provider_cache_stats())
# {"cross_provider_hits": {"anthropic→bedrock": 1}, "cost_saved_usd": 0.000015}
```

---

### 11. 💬 History Summarization — 60–80% off long conversations

When conversation exceeds N turns, uses Haiku to summarize older turns and keeps
recent turns verbatim. Prevents unbounded token growth in long sessions.

```python
client = OptimizedClient(
    anthropic_client=anthropic.Anthropic(),
    enable_history_summarization=True,
    history_summarization_threshold=10,  # summarize when > 10 turns
    history_keep_recent=4,               # always keep last 4 turns verbatim
)
```

---

### 12. ✍️ Prefill / Forced Output

Auto-detects required output format from system prompt keywords and injects an
assistant prefill to force structured responses — no extra system prompt tokens needed.

```python
client = OptimizedClient(
    anthropic_client=anthropic.Anthropic(),
    enable_prefill=True,
)

# System contains "json" → prefill '{"' auto-injected
response = client.complete(
    messages=[{"role": "user", "content": "Extract name and age from: John, 30"}],
    system="Always respond with valid JSON only.",
)

# Or explicit prefill
response = client.complete(messages=[...], prefill="```python\n")
```

**Auto-detected formats:** JSON (`"json"`, `"valid json"`), numbered list (`"enumerate"`),
code block (`"write code"`, `"implement"`).

---

### 13. 🎯 Dynamic Few-Shot Selector — 97% fewer few-shot tokens

Picks the top-K most relevant examples for each query using TF-IDF scoring instead of
sending all examples on every request.

```python
from llm_optimizer import FewShotSelector

selector = FewShotSelector(top_k=3)
selector.add_bulk([
    ("classify email as spam", "SPAM"),
    ("is this message promotional", "SPAM"),
    ("translate hello to french", "Bonjour"),
    ("what is the capital of Germany", "Berlin"),
])

examples = selector.select("categorize this email as unwanted")
formatted = selector.format(examples)
# Returns only the spam-related examples — not the translation or geography ones

system = f"Classify emails.\n\nExamples:\n{formatted}"
```

**Savings:** 100 examples × 50 tokens = 5,000 tokens hardcoded. Top 3 × 50 = 150 tokens. **97% reduction.**

---

### 14. 🔧 JSON Auto-Correction

Intercepts malformed JSON responses and fixes them — extraction first (free), then
Haiku as a fallback (~$0.001 per fix). No exception raised, no retry loop needed.

```python
client = OptimizedClient(
    anthropic_client=anthropic.Anthropic(),
    enable_output_correction=True,
)

# Model returns: 'Sure! Here is the result: {"name": "John", "age": 30}'
# Library extracts and returns: {"name": "John", "age": 30}
# Your code never sees the parse error

print(client.output_correction_stats())
# {"corrections_made": 1, "corrections_failed": 0, "success_rate_pct": 100.0}
```

---

### 15. 🌐 AWS Bedrock Support

Full support for Claude models via AWS Bedrock with automatic cachePoint translation.

```python
from llm_optimizer import OptimizedClient, Provider
from llm_optimizer.bedrock import make_bedrock_client

# Option 1 — AWS profile (local dev)
bedrock = make_bedrock_client(
    profile_name="my-profile",
    region="us-east-1",
    discount_pct=16.0,       # apply your negotiated AWS discount to cost estimates
    enable_caching=True,     # auto-translates cache_control → Bedrock cachePoint
    cache_ttl="1h",          # 5m or 1h (Claude 4.5 models only)
)

# Option 2 — env vars / IAM role (CI/CD, Lambda, EC2)
bedrock = make_bedrock_client()

client = OptimizedClient(
    bedrock_client=bedrock,
    preferred_provider=Provider.BEDROCK,
)
```

**Bedrock caching — what's different from direct API:**

| | Direct Anthropic | AWS Bedrock |
|---|---|---|
| Cache format | `cache_control` | `cachePoint` — auto-translated |
| Automatic caching | ✅ Supported | ❌ Not supported |
| Min cacheable tokens | 1,024 | 4,096 |
| TTL options | 5m or 1h | 5m or 1h |

The library translates `cache_control` → `cachePoint` and preserves TTL correctly.
LiteLLM has an open bug that strips TTL on Bedrock (GitHub #20326, Feb 2026) — this library does not.

---

### 16. 📡 Streaming Analytics

Real-time cost events per chunk during streaming. SSE-ready for web UIs that need
to show cost accumulating as tokens arrive.

```python
def on_token(event):
    print(f"Token #{event.token_count} | Cost so far: ${event.estimated_cost:.6f}")

with client.stream(
    messages=[{"role": "user", "content": "Write a poem about clouds"}],
    system="You are a poet.",
) as stream:
    for chunk in stream:
        print(chunk, end="", flush=True)

# Or wrap any stream with analytics
from llm_optimizer import wrap_stream_with_analytics

gen, collector = wrap_stream_with_analytics(
    stream_iter=your_stream,
    model_key="claude-sonnet-4",
    input_tokens=500,
    on_token_event=lambda e: send_sse_event(e),
    emit_every_n=5,
)

for chunk in gen:
    print(chunk, end="")

final = collector.finalize()
print(f"Total cost: ${final.estimated_cost:.6f} | Speed: {final.tokens_per_second:.1f} tok/s")
```

---

## Streaming

Use `client.stream()` for user-facing applications. `client.complete()` blocks
until the full response arrives.

```python
with client.stream(
    messages=[{"role": "user", "content": "Write a poem about clouds"}],
    system="You are a poet.",
    max_tokens=512,
) as stream:
    for text_chunk in stream:
        print(text_chunk, end="", flush=True)

usage = stream.usage()
if usage:
    print(f"\nTokens: {usage['input_tokens']} in, {usage['output_tokens']} out")
```

All optimizations apply identically to `stream()` and `complete()`.

**Provider support:**
- ✅ Anthropic (Claude)
- ✅ OpenAI (GPT)
- ✅ AWS Bedrock (Claude via Bedrock)
- ❌ Google (Gemini) — not yet implemented

---

## Error Handling and Fallback Behavior

| Failure | Behavior |
|---|---|
| Rate limit (429) | Retries with exponential backoff up to `max_retries` (default: 3) |
| Model overloaded (529) | Falls back to next cheapest capable model; emits `UserWarning` |
| Budget exceeded | `BudgetExceededError` raised before API call — request never sent |
| Exhausted retries | Raises the original provider exception |
| Non-retriable error | Raises immediately — no retry waste |
| LLM classifier failure | Silently falls back to heuristic routing |
| Cache miss | Transparent — proceeds normally |
| Batch API error | Re-queues requests; raises `RuntimeError` after `max_retries` |

```python
client = OptimizedClient(
    anthropic_client=anthropic.Anthropic(),
    max_retries=3,
    retry_base_delay=1.0,   # 1s → 2s → 4s
)
```

---

## Limitations

**Model router — heuristic mode:**
Keyword heuristics misclassify intent-heavy short prompts. A prompt like
`"Explain the constitutional implications of this clause"` is 8 words but requires
expert-level reasoning. Use `enable_llm_classifier=True` to avoid this.

**Document compression:**
All strategies are lossy. No semantic guarantee about what is preserved.
Do not use for legal, compliance, or code without prior validation.

**Streaming — cost tracking:**
`client.stream()` records estimated token counts at call time. Exact counts
are available via `stream.usage()` only after the stream is fully consumed.

**Token counting without tiktoken:**
The default heuristic estimator is within ~20–30% of actual counts for English prose.
Install `tiktoken` for exact counts: `pip install llm-optimizer[tiktoken]`.

**Pricing staleness:**
Prices are read from `pricing.json`. The library warns if data is older than 30 days.
Verify estimates against provider pricing pages before financial decisions.

**Async support:**
`client.complete()` and `client.stream()` are synchronous. Async support is planned
for a future release.

**OpenAI and Google — partially tested:**
Both are implemented but not tested end-to-end in CI. Anthropic and Bedrock are
the primary tested providers.

**Enterprise features:**
PII masking, compliance hashing, multi-tenant budgets, ROI reporting, cross-provider
failover, rate limiting, corporate memory, and the enterprise proxy are available
in `llm-optimizer-enterprise` — the companion package for corporate governance
and compliance requirements.

---

## Configuration Reference

```python
client = OptimizedClient(
    # Provider clients
    anthropic_client=anthropic.Anthropic(),
    openai_client=openai.OpenAI(),            # optional
    google_client=genai.GenerativeModel(),    # optional
    bedrock_client=make_bedrock_client(),     # optional

    # Core optimizations
    enable_caching=True,            # Anthropic cache_control injection
    enable_routing=True,            # auto model selection
    enable_optimization=True,       # prompt filler stripping
    enable_compression=False,       # ⚠️ lossy — opt-in only
    enable_tracking=True,           # cost analytics
    enable_batching=False,          # Anthropic Batch API

    # Routing
    preferred_provider=Provider.ANTHROPIC,
    default_model=None,             # None = auto-route
    enable_llm_classifier=False,    # accurate routing; ~$0.000015/call

    # Prompt optimization
    aggressive_optimization=False,
    max_history_turns=10,

    # Compression
    max_document_tokens=4000,
    compression_strategy="smart",   # "extractive" | "smart" | "truncate"

    # Budget enforcement
    max_cost_per_request=None,      # USD per request
    monthly_budget=None,            # USD per calendar month
    total_budget=None,              # USD lifetime
    on_budget_warning=None,         # callback(pct_used, budget_type)
    persist_budget=None,            # path to persist budget state

    # Output cache
    enable_output_cache=False,
    output_cache_backend="memory",  # "memory" | "sqlite"
    output_cache_ttl=3600,
    output_cache_max_size=1000,
    output_cache_path="llm_cache.db",

    # Semantic cache
    enable_semantic_cache=False,
    semantic_cache_threshold=0.92,
    semantic_cache_ttl=3600,

    # Cross-provider cache
    enable_cross_provider_cache=False,
    cross_provider_cache_ttl=3600,

    # History summarization
    enable_history_summarization=False,
    history_summarization_threshold=10,
    history_keep_recent=4,

    # Prefill / forced output
    enable_prefill=False,

    # JSON auto-correction
    enable_output_correction=False,

    # Streaming analytics
    streaming_analytics_callback=None,

    # Cost tracking
    persist_tracking="usage.jsonl",

    # Reliability
    max_retries=3,
    retry_base_delay=1.0,

    # Batching
    batch_default_model=None,
)
```

---

## Pricing Data

Prices are stored in `llm_optimizer/pricing.json` — not hardcoded in source.

```python
from llm_optimizer.models import pricing_info
print(pricing_info())
# {"version": "2026-07-14", "last_updated": "2026-07-14", "sources": {...}}
```

The library emits a `UserWarning` automatically if pricing data is older than 30 days.
Update by editing `pricing.json` and bumping `last_updated`.

---

## Available Models

> ⚠️ Prices as of `pricing.json` version `2026-07-14`. Verify at provider pricing pages.

| Key | Model | Provider | Input/1M | Cached/1M | Output/1M | Complexity |
|---|---|---|---|---|---|---|
| `claude-haiku-4-5` | Claude Haiku 4.5 | Anthropic | $1.00 | $0.10 | $5.00 | Simple, Medium |
| `claude-haiku-3-5` | Claude Haiku 3.5 | Anthropic | $0.80 | $0.08 | $4.00 | Simple, Medium |
| `claude-sonnet-4` | Claude Sonnet 4.5 | Anthropic | $3.00 | $0.30 | $15.00 | Simple–Complex |
| `claude-opus-4` | Claude Opus 4.5 | Anthropic | $5.00 | $0.50 | $25.00 | All |
| `bedrock-claude-haiku-4-5` | Claude Haiku 4.5 (Bedrock) | AWS Bedrock | $0.80 | $0.08 | $4.00 | Simple, Medium |
| `bedrock-claude-sonnet-4-5` | Claude Sonnet 4.5 (Bedrock) | AWS Bedrock | $3.00 | $0.30 | $15.00 | Simple–Complex |
| `gpt-5-mini` | GPT-5 Mini | OpenAI | $0.25 | $0.025 | $2.00 | Simple, Medium |
| `gpt-5` | GPT-5 | OpenAI | $2.50 | $0.25 | $15.00 | All |
| `gemini-flash` | Gemini 2.5 Flash | Google | $0.30 | $0.03 | $2.50 | Simple, Medium |
| `gemini-pro` | Gemini 2.5 Pro | Google | $1.25 | $0.125 | $10.00 | All |

---

## Test Suite

```bash
# Unit tests only — no API keys needed
pip install pytest
python -m pytest tests/ -v -m "not integration"

# Integration tests — requires ANTHROPIC_API_KEY
python -m pytest tests/ -v -m integration
```

**What the 172 unit tests cover (no real API calls):**
- Token estimation logic
- Model pricing calculations and cost math
- Heuristic routing decisions and regression cases
- LLM classifier routing (mocked Anthropic client)
- Prompt optimization transforms
- Document compression — all three strategies including TF-IDF scoring
- Cache_control block injection and Bedrock cachePoint translation
- TTL preservation on Bedrock (LiteLLM bug regression test)
- Budget enforcement — per-request, monthly, total; persistence
- Output cache — memory and SQLite backends, TTL expiry, LRU eviction
- Semantic cache — cosine similarity scoring, threshold, TTL
- Cross-provider cache — provider-agnostic keys, hit tracking
- Prefill injection and auto-detection from system prompt keywords
- Few-shot selector — TF-IDF relevance scoring, label filtering
- JSON auto-correction — extraction strategies and Haiku fallback
- Streaming analytics — chunk recording and finalize()
- Cost tracker arithmetic, persistence, thread safety
- Retry and fallback logic (mocked exceptions)

**What unit tests do NOT cover (requires real API):**
- Actual cache hits and cache write tokens from Anthropic
- LLM classifier calls against live API
- Batch submission, polling, and result collection
- End-to-end OpenAI and Google provider calls
- Streaming token delivery timing
- Bedrock IAM authentication

---

## Project Structure

```
llm_optimizer/
├── __init__.py             Public API surface
├── client.py               OptimizedClient — main entry point
│
├── COST REDUCTION
│   ├── caching.py          Anthropic cache_control injection
│   ├── bedrock_cache.py    Bedrock cachePoint translation
│   ├── router.py           Heuristic + LLM classifier routing
│   ├── optimizer.py        Prompt optimization + TF-IDF compression
│   ├── batcher.py          Anthropic Batch API wrapper
│   └── history.py          Conversation history summarization
│
├── CACHING LAYER
│   ├── cache.py            Output cache — memory + SQLite
│   ├── semantic_cache.py   Fuzzy cache — TF-IDF cosine similarity
│   └── cross_provider_cache.py  Cross-provider shared cache
│
├── PROMPT ENGINEERING
│   ├── prefill.py          Forced output — assistant prefill injection
│   ├── few_shot.py         Dynamic few-shot selector
│   └── output_repair.py    JSON auto-correction
│
├── PROVIDER INTEGRATIONS
│   ├── bedrock.py          AWS Bedrock client wrapper
│   └── streaming.py        Streaming — Anthropic, OpenAI, Bedrock
│
├── OBSERVABILITY
│   ├── tracker.py          Cost tracking + JSONL persistence
│   ├── budget.py           Budget enforcement
│   └── analytics.py        Streaming analytics — SSE-ready
│
├── CONFIG
│   ├── models.py           Model registry — loads from pricing.json
│   ├── pricing.json        Externalized pricing with version stamp
│   └── cli.py              llm-optimizer check / pricing-info

examples/
├── 01_basic_usage.py
├── 02_document_analysis.py
└── 03_batch_processing.py

tests/
└── test_llm_optimizer.py   172 unit tests, all mocked
```

---

## Changelog

### v0.4.2

- Sanitize docs, examples, and tests — replace org-specific AWS profile names with generic placeholders; Bedrock integration tests read `AWS_PROFILE` from the environment

### v0.4.1

- BUG FIX

### v0.4.0 — Major Feature Release

**Cost Reduction**
- Budget enforcement — `BudgetExceededError` raised before API call; zero cost on blocked requests
- Output cache — memory (LRU) and SQLite backends with configurable TTL and max size
- History summarization — Haiku-based compression of old conversation turns; 60–80% savings on long conversations

**Caching Layer**
- Semantic fuzzy cache — TF-IDF cosine similarity matching; catches paraphrases at zero cost
- Cross-provider cache — provider-agnostic keys shared across Anthropic, Bedrock, and OpenAI
- `cache_stats()`, `semantic_cache_stats()`, `cross_provider_cache_stats()` for observability

**Prompt Engineering**
- Prefill injection — auto-detects JSON/list/code from system prompt keywords; injects assistant prefill to force structured output
- Dynamic few-shot selector — top-K example selection by TF-IDF relevance; 97% fewer few-shot tokens vs hardcoded examples
- JSON auto-correction — extraction-first, falls back to Haiku for repair; eliminates try/except retry loops

**Streaming**
- Streaming analytics — per-chunk cost events, SSE-ready, `finalize()` for total cost and tokens/sec

**AWS Bedrock**
- Bedrock cachePoint translation — auto-translates Anthropic `cache_control` → Bedrock `cachePoint` format
- TTL preservation on Bedrock — fixes open LiteLLM bug (GitHub #20326, Feb 2026)
- `make_bedrock_client()` accepts `enable_caching` and `cache_ttl` parameters
- Correct model IDs matching Bedrock account format (`anthropic.claude-haiku-4-5-20251001-v1:0`)
- Negotiated discount support via `discount_pct`

**Test suite: 172 tests — all unit tests, no real API calls required**

> **Note:** PII masking, compliance hashing, multi-tenant budgets, ROI reporting,
> cross-provider failover, rate limiting, corporate memory, and the enterprise proxy
> are available in `llm-optimizer-enterprise` — the companion package for
> corporate governance and compliance requirements.

---

### v0.3.1

- GitHub repository URL corrected in `pyproject.toml`
- Bedrock model IDs updated to correct AWS ARN format (`anthropic.claude-*` not `us.anthropic.claude-*`)

---

### v0.3.0

- AWS Bedrock support — `BedrockClient` and `make_bedrock_client()` factory
- Both auth patterns — AWS profile (`profile_name="my-profile"`) and env vars / IAM role
- Bedrock pricing added to `pricing.json` with negotiated discount support
- `Provider.BEDROCK` added, Bedrock entries in `TIER_DEFAULTS` for all complexity tiers
- Note: Bedrock cachePoint translation was added in v0.4.0; this version stripped `cache_control` silently

---

### v0.2.0

- Streaming via `client.stream()` for Anthropic and OpenAI
- LLM classifier routing — opt-in via `enable_llm_classifier=True`
- `router.explain()` — full audit trail of routing decisions
- TF-IDF extractive document compression strategy
- Retry with exponential backoff on rate limits (429)
- Model fallback on overload errors (529)
- Pricing externalized to `pricing.json` with staleness warning after 30 days
- Test suite expanded from 36 to 62 tests

---

### v0.1.0 — Initial Release

- Prompt caching with automatic `cache_control` injection
- Heuristic model routing by keyword and token count
- Prompt optimization — filler phrase removal, whitespace normalization
- Batch processing via Anthropic Message Batch API
- Document compression — smart and truncate strategies
- Cost tracking with JSONL persistence
- 36 unit tests

---

## License

MIT