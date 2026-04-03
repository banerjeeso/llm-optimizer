"""
Example 1: Basic usage with Anthropic Claude

Demonstrates:
  - OptimizedClient setup
  - Automatic model routing
  - Prompt caching
  - Cost tracking
"""

# pip install anthropic llm-optimizer
import anthropic
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_optimizer import OptimizedClient, TaskComplexity


def main():
    # Setup — wrap your existing anthropic client
    raw_client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env
    client = OptimizedClient(
        anthropic_client=raw_client,
        enable_caching=True,
        enable_routing=True,
        enable_optimization=True,
    )

    # ─── Example 1: Simple classification (auto-routed to cheapest model) ─────
    print("--- Simple Task (auto-routes to Haiku) ---")
    response = client.complete(
        messages=[{"role": "user", "content": "Classify this text as positive/negative/neutral: 'Great product!'"}],
        system="You are a sentiment classifier. Reply with only the label.",
        tags={"task": "sentiment"},
    )
    print("Result:", response.content[0].text)

    # ─── Example 2: Complex task (auto-routes to Sonnet) ──────────────────────
    print("\n--- Complex Task (auto-routes to Sonnet) ---")
    response = client.complete(
        messages=[{"role": "user", "content": "Write a Python function that implements binary search with O(log n) complexity. Include type hints and docstring."}],
        system="You are an expert Python developer.",
        complexity=TaskComplexity.COMPLEX,
        max_tokens=512,
        tags={"task": "code_gen"},
    )
    print("Result (first 200 chars):", response.content[0].text[:200])

    # ─── Example 3: Pre-estimate cost before sending ──────────────────────────
    print("\n--- Cost Estimate (before sending) ---")
    estimate = client.estimate_cost(
        messages=[{"role": "user", "content": "Analyze this 10,000 word essay..."}],
        system="You are an editor.",
    )
    print(f"Estimated cost: ${estimate['estimated_cost_usd']:.6f}")
    print(f"With caching:   ${estimate['cached_cost_usd']:.6f}")
    print(f"Cache savings:  ${estimate['cache_savings_usd']:.6f}")

    # ─── Print usage summary ──────────────────────────────────────────────────
    client.print_summary()


if __name__ == "__main__":
    main()
