# llm-optimizer
Reduce LLM API costs by up to 90% — a Python library that wraps Anthropic, OpenAI, and Google clients and transparently applies six cost-saving techniques.


```python
import anthropic
from llm_optimizer import OptimizedClient

client = OptimizedClient(anthropic_client=anthropic.Anthropic())

response = client.complete(
    messages=[{"role": "user", "content": "Summarize this article..."}],
    system="You are a helpful assistant.",
)

client.print_summary()
# ═══════════════════════════════════════════════
#   💰 LLM Cost Optimizer — Usage Summary
# ═══════════════════════════════════════════════
#   Total Requests   : 10
#   Total Cost       : $0.000842
#   Total Saved      : $0.007580  (90% savings)
```

---

## Six Optimization Techniques

| # | Technique | Savings | How it works |
|---|-----------|---------|--------------|
| 1 | **Prompt Caching** | up to 90% | Injects `cache_control` breakpoints so repeated system prompts / documents are served from cache |
| 2 | **Model Routing** | 60–80% | Auto-selects the cheapest model capable of handling your task (Haiku for simple tasks, Sonnet/Opus for complex) |
| 3 | **Prompt Optimization** | 5–20% | Strips filler phrases, redundant whitespace, and verbose constructs from prompts |
| 4 | **Batch Processing** | 50% | Queues non-urgent requests and submits them via Anthropic's Batch API |
| 5 | **Document Compression** | variable | Smart-truncates large documents before sending, prioritizing relevant sections |
| 6 | **Cost Tracking** | — | Full usage analytics so you can measure and report savings over time |

---

## Installation

```bash
pip install llm-optimizer

# With exact token counting (recommended)
pip install llm-optimizer[tiktoken]

# All providers + exact token counting
pip install llm-optimizer[all]
```

---

## Quick Start

### Basic Usage

```python
import anthropic
from llm_optimizer import OptimizedClient

client = OptimizedClient(
    anthropic_client=anthropic.Anthropic(),  # Your existing client
    enable_caching=True,        # Prompt caching (default: True)
    enable_routing=True,        # Model routing (default: True)
    enable_optimization=True,   # Prompt optimization (default: True)
    enable_compression=True,    # Document compression (default: True)
    enable_tracking=True,       # Cost tracking (default: True)
)

# Use just like a normal client — optimizations happen automatically
response = client.complete(
    messages=[{"role": "user", "content": "Is this email spam? 'You won a prize!'"}],
    system="You are a spam classifier. Reply with SPAM or NOT SPAM.",
)
print(response.content[0].text)  # → NOT SPAM

client.print_summary()
```

### Technique 1: Prompt Caching

Cache stable content (system prompts, documents, few-shot examples) so it's only charged once:

```python
from llm_optimizer import OptimizedClient, build_cached_system_prompt

client = OptimizedClient(anthropic_client=anthropic.Anthropic())

# Build a system prompt with automatic cache breakpoints
# Most stable content first → least stable last
system = client.build_cached_system(
    base_instructions="""
    You are an expert legal analyst with 20 years of experience...
    [your long stable instructions here]
    """,  # ← cached (charged once, reused forever)

    few_shot_examples="""
    Example 1: Contract analysis...
    Example 2: Risk assessment...
    """,  # ← cached separately

    knowledge_base="""
    [Large reference document, 50K tokens]
    """,  # ← also cached
)

# First call: pays full price for the system prompt
response1 = client.complete(messages=[...], system=system)

# Second+ calls: system prompt served from cache at 10% of cost
response2 = client.complete(messages=[...], system=system)  # 90% cheaper!
```

### Technique 2: Model Routing

Automatically use the cheapest capable model:

```python
from llm_optimizer import OptimizedClient, TaskComplexity

client = OptimizedClient(anthropic_client=anthropic.Anthropic())

# Simple task → automatically routed to Claude Haiku ($0.001/1K tokens)
response = client.complete(
    messages=[{"role": "user", "content": "Translate 'Hello' to French"}],
    # No model specified → router chooses cheapest capable model
)

# Override if you know the complexity
response = client.complete(
    messages=[{"role": "user", "content": "Design a microservices architecture..."}],
    complexity=TaskComplexity.EXPERT,  # → Claude Opus
)

# See routing decision before calling
from llm_optimizer import ModelRouter, Provider
router = ModelRouter(preferred_provider=Provider.ANTHROPIC)
savings = router.estimate_savings(
    "classify this email as spam or not",
    expensive_model_key="claude-opus-4"
)
# → {"savings_pct": 80.0, "savings_usd": 0.0000040, "recommended_model": "claude-haiku-4-5"}
```

### Technique 3: Prompt Optimization

```python
from llm_optimizer import PromptOptimizer

optimizer = PromptOptimizer(enabled=True, aggressive=False)

result = optimizer.optimize("""
    In order to complete this task, please make sure to carefully analyze
    the following text. It is important to note that you should classify
    it as positive, negative, or neutral.   Please be aware that
    accuracy   is important.
""")

print(result.optimized_text)
# → "To complete this task, carefully analyze the following text.
#    Classify it as positive, negative, or neutral.
#    Accuracy is important."

print(f"Saved {result.tokens_saved} tokens ({result.savings_pct}%)")
```

### Technique 4: Batch Processing

```python
import anthropic
from llm_optimizer import OptimizedClient

client = OptimizedClient(
    anthropic_client=anthropic.Anthropic(),
    enable_batching=True,  # Must opt-in
)

# Queue 1000s of requests — they'll be processed at 50% cost
articles = load_articles_from_db()  # your data
for article in articles:
    client.queue(
        custom_id=f"article-{article['id']}",
        messages=[{"role": "user", "content": f"Summarize: {article['text']}"}],
        max_tokens=200,
    )

# Submit as a single batch (50% cheaper, results in minutes-hours)
batch_id = client.submit_batch()

# Poll when ready
results = client.poll_batch(batch_id, wait=True, timeout_seconds=3600)
for r in results:
    print(f"{r.custom_id}: {r.content}")
```

### Technique 5: Document Compression

```python
from llm_optimizer import OptimizedClient

client = OptimizedClient(
    anthropic_client=anthropic.Anthropic(),
    max_document_tokens=4000,  # Compress any document over 4K tokens
)

# Pass raw long documents — library compresses before sending
response = client.complete(
    messages=[{"role": "user", "content": "What are the key risks?"}],
    documents=[very_long_contract],   # Compressed to 4K tokens automatically
    system="You are a legal analyst.",
)

# Or compress manually
compressed, tokens_saved = client.compress_document(
    document=long_doc,
    query="payment terms",  # Focus compression on relevant sections
)
print(f"Reduced from {original_tokens} to {estimate_tokens(compressed)} tokens")
```

### Technique 6: Cost Tracking

```python
client = OptimizedClient(
    anthropic_client=anthropic.Anthropic(),
    persist_tracking="usage.jsonl",  # Persist across restarts
)

# ... run your application ...

# Get full summary
summary = client.summary()
print(f"Total cost: ${summary['total_cost_usd']:.4f}")
print(f"Savings:    ${summary['total_cost_saved_usd']:.4f} ({summary['savings_pct']}%)")
print(f"By model:   {summary['by_model']}")

# Pretty print
client.print_summary()
```

---

## Estimate Cost Before Calling

```python
estimate = client.estimate_cost(
    messages=[{"role": "user", "content": large_prompt}],
    system="You are a helpful assistant.",
)

if estimate["estimated_cost_usd"] > 0.01:
    print("Warning: expensive request!")
    # Route to batch, compress, or simplify
```

---

## Configuration Reference

```python
client = OptimizedClient(
    # Provider clients
    anthropic_client=anthropic.Anthropic(),
    openai_client=openai.OpenAI(),          # Optional
    google_client=genai.GenerativeModel(),  # Optional

    # Feature flags
    enable_caching=True,
    enable_routing=True,
    enable_optimization=True,
    enable_compression=True,
    enable_tracking=True,
    enable_batching=False,          # Opt-in (async workflow)

    # Routing
    preferred_provider=Provider.ANTHROPIC,
    default_model=None,             # None = auto-route

    # Optimization
    aggressive_optimization=False,  # More aggressive prompt trimming
    max_history_turns=10,           # Trim conversation history after N turns

    # Compression
    max_document_tokens=4000,       # Compress docs over this size

    # Tracking
    persist_tracking="usage.jsonl", # Path to persist usage data
)
```

---

## Available Models

| Key | Model | Provider | Input/1M | Cached/1M | Output/1M | Best For |
|-----|-------|----------|----------|-----------|-----------|----------|
| `claude-haiku-4-5` | Claude Haiku 4.5 | Anthropic | $1.00 | $0.10 | $5.00 | Simple tasks |
| `claude-haiku-3-5` | Claude Haiku 3.5 | Anthropic | $0.80 | $0.08 | $4.00 | Simple tasks |
| `claude-sonnet-4` | Claude Sonnet 4.5 | Anthropic | $3.00 | $0.30 | $15.00 | Medium/Complex |
| `claude-opus-4` | Claude Opus 4.5 | Anthropic | $5.00 | $0.50 | $25.00 | Expert tasks |
| `gpt-5-mini` | GPT-5 Mini | OpenAI | $0.25 | $0.025 | $2.00 | Simple tasks |
| `gpt-5` | GPT-5 | OpenAI | $2.50 | $0.25 | $15.00 | Complex tasks |
| `gemini-flash` | Gemini 2.5 Flash | Google | $0.30 | $0.03 | $2.50 | Simple/Medium |
| `gemini-pro` | Gemini 2.5 Pro | Google | $1.25 | $0.125 | $10.00 | Complex tasks |

---

## Project Structure

```
llm_optimizer/
├── __init__.py         # Public API
├── client.py           # OptimizedClient — main entry point
├── models.py           # Model registry + pricing
├── router.py           # Intelligent model routing
├── optimizer.py        # Prompt optimization + document compression
├── caching.py          # Prompt cache management
├── batcher.py          # Batch processing (50% savings)
└── tracker.py          # Cost tracking + analytics

examples/
├── 01_basic_usage.py
├── 02_document_analysis.py
└── 03_batch_processing.py

tests/
└── test_llm_optimizer.py   # 36 tests, all passing
```

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## License

MIT

