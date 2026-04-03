"""
Example 3: Batch processing for non-urgent tasks (50% cost savings)

Demonstrates:
  - Queuing requests for batch processing
  - Submitting a batch to Anthropic
  - Polling for results
  - Cost comparison: live vs batch
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import anthropic
from llm_optimizer import OptimizedClient, MODELS


def main():
    raw_client = anthropic.Anthropic()
    client = OptimizedClient(
        anthropic_client=raw_client,
        enable_batching=True,
        enable_routing=True,
        enable_optimization=True,
        batch_default_model="claude-haiku-4-5-20251001",
    )

    # Sample items to process (e.g., product descriptions, support tickets, etc.)
    items = [
        ("item-001", "Our new wireless headphones feature 40-hour battery life."),
        ("item-002", "The software crashed when I tried to export my project."),
        ("item-003", "I love how fast the checkout process is on your website!"),
        ("item-004", "The packaging was damaged but the product itself is fine."),
        ("item-005", "Please cancel my subscription immediately."),
    ]

    system = "Classify the sentiment as POSITIVE, NEGATIVE, or NEUTRAL. Reply with just the label."

    print(f"Queueing {len(items)} requests for batch processing...")
    print("Estimated savings vs live API: 50%\n")

    # Queue all requests
    for item_id, text in items:
        client.queue(
            custom_id=item_id,
            messages=[{"role": "user", "content": text}],
            system=system,
            max_tokens=10,
        )

    print(f"Queue size: {client.batcher.queue_size()} requests")

    # Submit the batch
    batch_id = client.submit_batch()
    print(f"Batch submitted: {batch_id}")

    if batch_id and not batch_id.startswith("dry_run"):
        # Poll for results (blocks until complete)
        print("\nWaiting for batch results...")
        results = client.poll_batch(
            batch_id,
            wait=True,
            timeout_seconds=3600,
            on_progress=lambda s: print(f"  Status: {s['status']}"),
        )

        print("\n--- Batch Results ---")
        for result in results:
            if result.success:
                print(f"  {result.custom_id}: {result.content}")
            else:
                print(f"  {result.custom_id}: ERROR — {result.error}")
    else:
        print("\n(Dry-run mode — no real API calls made)")
        print("In production, results arrive within minutes to hours.")
        print("Cost savings: ~50% vs real-time API calls")


def cost_comparison():
    """Show cost difference between live and batch processing."""
    model_key = "claude-haiku-4-5"
    model = MODELS[model_key]

    n_requests = 1000
    avg_input_tokens = 200
    avg_output_tokens = 50

    live_cost = model.estimate_cost(avg_input_tokens, avg_output_tokens) * n_requests
    # Batch is ~50% of live cost
    batch_cost = live_cost * 0.5

    print("\n--- Cost Comparison: Live vs Batch ---")
    print(f"Model: {model.display_name}")
    print(f"Requests: {n_requests:,}")
    print(f"Avg tokens per request: {avg_input_tokens} in + {avg_output_tokens} out")
    print(f"")
    print(f"Live API cost:  ${live_cost:.4f}")
    print(f"Batch API cost: ${batch_cost:.4f}")
    print(f"Savings:        ${live_cost - batch_cost:.4f} ({50}%)")


if __name__ == "__main__":
    cost_comparison()
    main()
