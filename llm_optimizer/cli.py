"""
CLI for llm-optimizer.

Commands:
  llm-optimizer pricing-info     Show current pricing data version and staleness
  llm-optimizer update-pricing   Instructions to update pricing.json
  llm-optimizer check            Verify install and show config
"""

import sys
import json
from pathlib import Path


def main():
    args = sys.argv[1:]
    cmd = args[0] if args else "check"

    if cmd == "pricing-info":
        _pricing_info()
    elif cmd == "update-pricing":
        _update_pricing_instructions()
    elif cmd == "check":
        _check()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: llm-optimizer [pricing-info | update-pricing | check]")
        sys.exit(1)


def _pricing_info():
    pricing_path = Path(__file__).parent / "pricing.json"
    with open(pricing_path) as f:
        data = json.load(f)
    meta = data.get("_meta", {})
    print(f"\nllm-optimizer pricing data")
    print(f"  Version      : {meta.get('version', 'unknown')}")
    print(f"  Last updated : {meta.get('last_updated', 'unknown')}")
    print(f"  Models       : {len(data.get('models', {}))}")
    print(f"\nSources to verify prices:")
    for provider, url in meta.get("sources", {}).items():
        print(f"  {provider:12}: {url}")
    print(f"\nWarning: {meta.get('warning', '')}\n")


def _update_pricing_instructions():
    pricing_path = Path(__file__).parent / "pricing.json"
    print(f"""
To update pricing data:

1. Open {pricing_path}
2. Verify prices against provider pages:
   - Anthropic : https://www.anthropic.com/pricing
   - OpenAI    : https://openai.com/pricing
   - Google    : https://ai.google.dev/pricing
3. Update the relevant model prices
4. Bump "_meta.last_updated" to today's date (YYYY-MM-DD)
5. Bump "_meta.version" to match

The library will stop warning about staleness once last_updated is within 30 days.
""")


def _check():
    try:
        from llm_optimizer import __version__
        from llm_optimizer.models import pricing_info, MODELS
        info = pricing_info()
        print(f"\nllm-optimizer v{__version__} installed ✓")
        print(f"  Models loaded  : {len(MODELS)}")
        print(f"  Pricing version: {info['last_updated']}")
        print(f"\nRun 'llm-optimizer pricing-info' for full pricing details.\n")
    except Exception as e:
        print(f"Install check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
