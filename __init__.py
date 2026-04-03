"""
llm_optimizer — Reduce LLM API costs by up to 90%

Six automatic optimizations:
  1. Prompt caching       — reuse repeated content (up to 90% savings)
  2. Model routing        — cheapest capable model for each task
  3. Prompt optimization  — reduce token count without losing meaning
  4. Batch processing     — 50% savings on non-urgent requests
  5. Document compression — smart truncation before sending
  6. Cost tracking        — full usage & savings analytics

Quick start:
    import anthropic
    from llm_optimizer import OptimizedClient

    client = OptimizedClient(anthropic_client=anthropic.Anthropic())

    response = client.complete(
        messages=[{"role": "user", "content": "Summarize this article: ..."}],
        system="You are a helpful assistant.",
    )

    client.print_summary()
"""

from .client import OptimizedClient
from .models import ModelConfig, Provider, TaskComplexity, MODELS
from .router import ModelRouter
from .optimizer import PromptOptimizer, DocumentCompressor, OptimizationResult
from .caching import CacheManager, build_cached_system_prompt
from .batcher import BatchProcessor, BatchRequest, BatchResult
from .tracker import CostTracker, UsageRecord, estimate_tokens

__version__ = "0.1.0"
__author__ = "Your Team"

__all__ = [
    # Main client
    "OptimizedClient",
    # Models & routing
    "ModelConfig",
    "Provider",
    "TaskComplexity",
    "MODELS",
    "ModelRouter",
    # Optimization
    "PromptOptimizer",
    "DocumentCompressor",
    "OptimizationResult",
    # Caching
    "CacheManager",
    "build_cached_system_prompt",
    # Batching
    "BatchProcessor",
    "BatchRequest",
    "BatchResult",
    # Tracking
    "CostTracker",
    "UsageRecord",
    "estimate_tokens",
]
