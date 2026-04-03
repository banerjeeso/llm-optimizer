"""
Token counting utilities and cost/usage tracker.
"""

import re
import time
import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import ModelConfig, MODELS


# ─── Token Estimation ──────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    Fast token estimator without requiring tiktoken.
    Uses heuristics based on BPE tokenization patterns:
      - English prose   ≈ 1.3 tokens/word
      - Code/symbols    ≈ 2-3 tokens/word
      - CJK characters  ≈ 2 tokens/char

    For production use, install tiktoken for exact counts.
    """
    if not text:
        return 0

    # Try tiktoken first if available
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        pass

    # Heuristic fallback
    # Count CJK characters (they tokenize differently)
    cjk_count = len(re.findall(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))

    # Remove CJK for word counting
    latin_text = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', '', text)

    # Count code-like tokens (symbols, operators)
    code_symbols = len(re.findall(r'[{}()\[\]<>;:=+\-*/%&|^~!@#$,.]', latin_text))

    # Count words
    words = len(re.findall(r'\b\w+\b', latin_text))

    # Blended estimate
    estimated = (
        cjk_count * 2          # CJK ~2 tokens each
        + words * 1.3           # English words ~1.3 tokens
        + code_symbols * 0.5    # Extra for symbols
    )
    return max(1, int(estimated))


def count_messages_tokens(messages: list[dict]) -> int:
    """Estimate total tokens for a list of chat messages."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    total += estimate_tokens(block.get("text", ""))
        total += 4  # Per-message overhead (role, separators)
    return total + 2  # Conversation overhead


# ─── Usage Record ──────────────────────────────────────────────────────────────

@dataclass
class UsageRecord:
    timestamp: str
    model_key: str
    model_id: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float
    request_type: str           # "live" | "batch" | "cached"
    optimizations_applied: list[str] = field(default_factory=list)
    tokens_saved: int = 0
    cost_saved: float = 0.0
    latency_ms: Optional[float] = None
    tags: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Cost Tracker ──────────────────────────────────────────────────────────────

class CostTracker:
    """
    Thread-safe tracker for LLM usage and cost.
    Optionally persists records to a JSONL file.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._records: list[UsageRecord] = []
        self._lock = threading.Lock()
        self._persist_path = Path(persist_path) if persist_path else None

        if self._persist_path and self._persist_path.exists():
            self._load()

    def record(
        self,
        model_key: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        request_type: str = "live",
        optimizations_applied: Optional[list[str]] = None,
        tokens_saved: int = 0,
        latency_ms: Optional[float] = None,
        tags: Optional[dict] = None,
    ) -> UsageRecord:
        """Record a usage event and return the record."""
        model = MODELS.get(model_key)
        model_id = model.model_id if model else model_key

        cost = 0.0
        cost_saved = 0.0
        if model:
            cost = model.estimate_cost(input_tokens, output_tokens, cached_tokens)
            cost_saved = model.cache_savings(cached_tokens) if cached_tokens else 0.0

        rec = UsageRecord(
            timestamp=datetime.utcnow().isoformat(),
            model_key=model_key,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost,
            request_type=request_type,
            optimizations_applied=optimizations_applied or [],
            tokens_saved=tokens_saved,
            cost_saved=cost_saved,
            latency_ms=latency_ms,
            tags=tags or {},
        )

        with self._lock:
            self._records.append(rec)
            if self._persist_path:
                self._append_record(rec)

        return rec

    def summary(self) -> dict:
        """Return aggregated usage statistics."""
        with self._lock:
            if not self._records:
                return {"total_requests": 0, "total_cost_usd": 0.0}

            total_cost = sum(r.cost_usd for r in self._records)
            total_saved = sum(r.cost_saved for r in self._records)
            total_input = sum(r.input_tokens for r in self._records)
            total_output = sum(r.output_tokens for r in self._records)
            total_cached = sum(r.cached_tokens for r in self._records)
            total_tokens_saved = sum(r.tokens_saved for r in self._records)

            by_model: dict[str, dict] = {}
            for r in self._records:
                if r.model_key not in by_model:
                    by_model[r.model_key] = {"requests": 0, "cost": 0.0, "input_tokens": 0, "output_tokens": 0}
                by_model[r.model_key]["requests"] += 1
                by_model[r.model_key]["cost"] += r.cost_usd
                by_model[r.model_key]["input_tokens"] += r.input_tokens
                by_model[r.model_key]["output_tokens"] += r.output_tokens

            by_type = {}
            for r in self._records:
                by_type[r.request_type] = by_type.get(r.request_type, 0) + 1

            optimization_counts: dict[str, int] = {}
            for r in self._records:
                for opt in r.optimizations_applied:
                    optimization_counts[opt] = optimization_counts.get(opt, 0) + 1

            return {
                "total_requests": len(self._records),
                "total_cost_usd": round(total_cost, 6),
                "total_cost_saved_usd": round(total_saved, 6),
                "savings_pct": round(total_saved / (total_cost + total_saved) * 100, 1) if (total_cost + total_saved) > 0 else 0,
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_cached_tokens": total_cached,
                "total_tokens_saved": total_tokens_saved,
                "by_model": by_model,
                "by_request_type": by_type,
                "optimizations_applied": optimization_counts,
            }

    def print_summary(self):
        """Pretty-print usage summary to console."""
        s = self.summary()
        print("\n" + "=" * 55)
        print("  💰 LLM Cost Optimizer — Usage Summary")
        print("=" * 55)
        print(f"  Total Requests   : {s['total_requests']}")
        print(f"  Total Cost       : ${s['total_cost_usd']:.6f}")
        print(f"  Total Saved      : ${s['total_cost_saved_usd']:.6f}  ({s['savings_pct']}% savings)")
        print(f"  Input Tokens     : {s['total_input_tokens']:,}")
        print(f"  Output Tokens    : {s['total_output_tokens']:,}")
        print(f"  Cached Tokens    : {s['total_cached_tokens']:,}")
        print(f"  Tokens Saved     : {s['total_tokens_saved']:,}")
        if s.get("by_model"):
            print("\n  By Model:")
            for model, stats in s["by_model"].items():
                print(f"    {model}: {stats['requests']} reqs, ${stats['cost']:.6f}")
        if s.get("by_request_type"):
            print("\n  By Type:")
            for rtype, count in s["by_request_type"].items():
                print(f"    {rtype}: {count}")
        if s.get("optimizations_applied"):
            print("\n  Optimizations Used:")
            for opt, count in s["optimizations_applied"].items():
                print(f"    {opt}: {count}x")
        print("=" * 55 + "\n")

    def _append_record(self, rec: UsageRecord):
        with open(self._persist_path, "a") as f:
            f.write(json.dumps(rec.to_dict()) + "\n")

    def _load(self):
        with open(self._persist_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        self._records.append(UsageRecord(**data))
                    except Exception:
                        pass
