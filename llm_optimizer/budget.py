"""
Budget Guard — enforces cost limits before and during API calls.

Two types of limits:
  - Per-request limit: raises before sending if estimated cost exceeds threshold
  - Monthly/total budget: raises when cumulative spend crosses the budget

Usage:
    from llm_optimizer import OptimizedClient
    from llm_optimizer.budget import BudgetExceededError

    client = OptimizedClient(
        anthropic_client=...,
        max_cost_per_request=0.01,   # $0.01 per request max
        monthly_budget=50.00,        # $50/month total
        on_budget_warning=lambda pct: print(f"Budget {pct:.0f}% used"),
    )

    try:
        response = client.complete(messages=[...])
    except BudgetExceededError as e:
        print(f"Blocked: {e}")

Budget state persists to the same JSONL file as cost tracking
so it survives restarts. Reset with client.reset_budget().
"""

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Callable, Optional


class BudgetExceededError(Exception):
    """
    Raised before an API call when it would exceed a configured budget.
    The request is NOT sent — no cost is incurred.

    Attributes:
        limit_type:       "per_request" or "total_budget"
        estimated_cost:   What this request would have cost
        current_spend:    Total spent so far in this period
        budget:           The configured limit
    """
    def __init__(self, limit_type: str, estimated_cost: float,
                 current_spend: float, budget: float):
        self.limit_type = limit_type
        self.estimated_cost = estimated_cost
        self.current_spend = current_spend
        self.budget = budget
        super().__init__(
            f"Budget exceeded ({limit_type}): "
            f"estimated=${estimated_cost:.6f}, "
            f"spent=${current_spend:.6f}, "
            f"limit=${budget:.6f}"
        )


@dataclass
class BudgetState:
    total_spent: float = 0.0
    request_count: int = 0
    period_start: str = field(default_factory=lambda: date.today().isoformat())
    last_reset: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "total_spent": self.total_spent,
            "request_count": self.request_count,
            "period_start": self.period_start,
            "last_reset": self.last_reset,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BudgetState":
        return cls(
            total_spent=data.get("total_spent", 0.0),
            request_count=data.get("request_count", 0),
            period_start=data.get("period_start", date.today().isoformat()),
            last_reset=data.get("last_reset", datetime.utcnow().isoformat()),
        )


class BudgetGuard:
    """
    Enforces cost limits on LLM API calls.

    Checks happen BEFORE the API call — if the estimated cost would
    exceed any limit, BudgetExceededError is raised and no request is sent.

    Args:
        max_cost_per_request:  Max USD per single request. None = no limit.
        monthly_budget:        Max USD per calendar month. None = no limit.
        total_budget:          Max USD lifetime total. None = no limit.
        warning_threshold_pct: Call on_budget_warning when this % of budget used.
        on_budget_warning:     Callback(pct_used) when warning threshold crossed.
        persist_path:          File to persist budget state across restarts.
    """

    def __init__(
        self,
        max_cost_per_request: Optional[float] = None,
        monthly_budget: Optional[float] = None,
        total_budget: Optional[float] = None,
        warning_threshold_pct: float = 80.0,
        on_budget_warning: Optional[Callable[[float, str], None]] = None,
        persist_path: Optional[str] = None,
    ):
        self.max_cost_per_request = max_cost_per_request
        self.monthly_budget = monthly_budget
        self.total_budget = total_budget
        self.warning_threshold_pct = warning_threshold_pct
        self.on_budget_warning = on_budget_warning
        self._persist_path = Path(persist_path) if persist_path else None
        self._lock = threading.Lock()
        self._state = BudgetState()
        self._warning_fired: set[str] = set()

        if self._persist_path and self._persist_path.exists():
            self._load()

    def check(self, estimated_cost: float):
        """
        Check if a request with this estimated cost is allowed.

        Raises BudgetExceededError if any limit would be exceeded.
        Call this BEFORE making the API request.

        Args:
            estimated_cost: Estimated cost of the request in USD
        """
        with self._lock:
            # 1. Per-request limit
            if self.max_cost_per_request is not None:
                if estimated_cost > self.max_cost_per_request:
                    raise BudgetExceededError(
                        limit_type="per_request",
                        estimated_cost=estimated_cost,
                        current_spend=self._state.total_spent,
                        budget=self.max_cost_per_request,
                    )

            # 2. Monthly budget — reset if new month
            self._maybe_reset_monthly()
            if self.monthly_budget is not None:
                if self._state.total_spent + estimated_cost > self.monthly_budget:
                    raise BudgetExceededError(
                        limit_type="monthly_budget",
                        estimated_cost=estimated_cost,
                        current_spend=self._state.total_spent,
                        budget=self.monthly_budget,
                    )

            # 3. Total budget
            if self.total_budget is not None:
                if self._state.total_spent + estimated_cost > self.total_budget:
                    raise BudgetExceededError(
                        limit_type="total_budget",
                        estimated_cost=estimated_cost,
                        current_spend=self._state.total_spent,
                        budget=self.total_budget,
                    )

            # 4. Warning threshold
            self._maybe_fire_warning()

    def record(self, actual_cost: float):
        """Record actual cost after a successful request."""
        with self._lock:
            self._state.total_spent += actual_cost
            self._state.request_count += 1
            self._maybe_fire_warning()
            if self._persist_path:
                self._save()

    def reset(self):
        """Reset budget state — use at start of new billing period."""
        with self._lock:
            self._state = BudgetState()
            self._warning_fired.clear()
            if self._persist_path:
                self._save()

    def status(self) -> dict:
        """Return current budget status."""
        with self._lock:
            result = {
                "total_spent_usd": round(self._state.total_spent, 6),
                "request_count": self._state.request_count,
                "period_start": self._state.period_start,
            }
            if self.max_cost_per_request is not None:
                result["max_cost_per_request"] = self.max_cost_per_request
            if self.monthly_budget is not None:
                result["monthly_budget"] = self.monthly_budget
                result["monthly_remaining"] = round(
                    max(0.0, self.monthly_budget - self._state.total_spent), 6
                )
                result["monthly_used_pct"] = round(
                    self._state.total_spent / self.monthly_budget * 100, 1
                )
            if self.total_budget is not None:
                result["total_budget"] = self.total_budget
                result["total_remaining"] = round(
                    max(0.0, self.total_budget - self._state.total_spent), 6
                )
            return result

    def _maybe_reset_monthly(self):
        """Reset state if we've crossed into a new calendar month."""
        if self.monthly_budget is None:
            return
        current_month = date.today().replace(day=1).isoformat()
        if self._state.period_start < current_month:
            self._state = BudgetState(period_start=current_month)
            self._warning_fired.discard("monthly_budget")

    def _maybe_fire_warning(self):
        """Fire warning callback if threshold crossed."""
        if not self.on_budget_warning:
            return
        for budget_type, limit in [
            ("monthly_budget", self.monthly_budget),
            ("total_budget", self.total_budget),
        ]:
            if limit is None:
                continue
            pct = self._state.total_spent / limit * 100
            key = f"{budget_type}_{int(self.warning_threshold_pct)}"
            if pct >= self.warning_threshold_pct and key not in self._warning_fired:
                self._warning_fired.add(key)
                self.on_budget_warning(pct, budget_type)

    def _save(self):
        with open(self._persist_path, "w") as f:
            json.dump(self._state.to_dict(), f)

    def _load(self):
        try:
            with open(self._persist_path) as f:
                self._state = BudgetState.from_dict(json.load(f))
        except Exception:
            self._state = BudgetState()
