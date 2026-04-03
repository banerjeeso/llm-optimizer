"""
Example 2: Document analysis with compression + caching

Demonstrates:
  - build_cached_system_prompt for stable system prompts
  - Document compression before sending
  - Cost comparison before/after optimization
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import anthropic
from llm_optimizer import OptimizedClient, build_cached_system_prompt, estimate_tokens


# A long system prompt shared across many requests — perfect for caching
BASE_INSTRUCTIONS = """
You are an expert document analyst specializing in legal and financial documents.
Your role is to extract key information, identify risks, and provide clear summaries.

When analyzing documents:
- Extract key parties, dates, and amounts
- Identify important clauses and obligations
- Flag potential risks or unusual terms
- Provide a structured summary with clear sections

Always be precise and factual. Never invent information not present in the document.
Cite specific sections when referencing document content.
""" * 15  # Repeat to exceed 1024-token cache threshold

# Few-shot examples — also cache-worthy
FEW_SHOT_EXAMPLES = """
Example analysis format:

Document: Service Agreement
Parties: TechCorp Inc. (Service Provider), Acme Ltd. (Client)
Key Terms:
  - Duration: 12 months, auto-renewing
  - Payment: $5,000/month, net-30
  - Termination: 30-day notice required
Risks:
  - Auto-renewal clause (Section 4.2) — requires active opt-out
  - Liability cap at $10,000 may be insufficient
""" * 8


def main():
    raw_client = anthropic.Anthropic()
    client = OptimizedClient(
        anthropic_client=raw_client,
        enable_caching=True,
        enable_compression=True,
        max_document_tokens=3000,  # Compress docs to 3K tokens
    )

    # Build optimally-cached system prompt
    # This gets cached once and reused across all requests = big savings
    system_blocks = client.build_cached_system(
        base_instructions=BASE_INSTRUCTIONS,
        few_shot_examples=FEW_SHOT_EXAMPLES,
    )

    print(f"System prompt tokens: {estimate_tokens(BASE_INSTRUCTIONS + FEW_SHOT_EXAMPLES):,}")
    print(f"System blocks with cache_control: {len([b for b in system_blocks if 'cache_control' in b])}")

    # Simulate a large document (e.g., a contract)
    large_document = """
    SERVICE AGREEMENT

    This Service Agreement ("Agreement") is entered into as of January 1, 2026,
    between TechCorp Inc., a Delaware corporation ("Provider"), and Acme Ltd. ("Client").

    1. SERVICES
    Provider agrees to deliver software development services including but not limited to:
    web application development, API integration, database design, and technical consulting.

    2. PAYMENT TERMS
    Client shall pay Provider $8,500 per month, due within 30 days of invoice.
    Late payments accrue interest at 1.5% per month.

    3. INTELLECTUAL PROPERTY
    All work product created under this Agreement shall be owned by Client upon full payment.
    Provider retains rights to general methodologies and pre-existing tools.

    4. CONFIDENTIALITY
    Both parties agree to maintain strict confidentiality of proprietary information
    for a period of 3 years following termination of this Agreement.

    5. TERM AND TERMINATION
    This Agreement commences on the date above and continues for 12 months.
    Either party may terminate with 30 days written notice.
    """ * 20  # Make it large to trigger compression

    original_tokens = estimate_tokens(large_document)
    print(f"\nOriginal document tokens: {original_tokens:,}")

    # Compress document before sending
    compressed, saved = client.compress_document(large_document, query="payment terms and termination")
    print(f"Compressed to: {estimate_tokens(compressed):,} tokens (saved {saved:,})")

    # Send with optimization
    response = client.complete(
        messages=[{"role": "user", "content": f"Analyze this contract:\n\n{compressed}"}],
        system=system_blocks,
        max_tokens=512,
        tags={"task": "contract_analysis", "doc_type": "service_agreement"},
    )

    print("\n--- Analysis Result ---")
    print(response.content[0].text[:500])

    # Second request — system prompt is now cached!
    print("\n--- Second Request (system prompt cached = 90% cheaper) ---")
    response2 = client.complete(
        messages=[{"role": "user", "content": "What are the key payment terms?"}],
        system=system_blocks,   # Same system → served from cache
        max_tokens=256,
    )
    print(response2.content[0].text)

    client.print_summary()


if __name__ == "__main__":
    main()
