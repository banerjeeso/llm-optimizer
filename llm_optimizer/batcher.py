"""
Batch Processor — queue non-urgent requests and submit them as
Anthropic Message Batches (50% cost reduction).

Anthropic Batch API:
  - Up to 10,000 requests per batch
  - Results within 24 hours (usually <1 hour)
  - 50% cheaper than real-time API

Usage:
    batcher = BatchProcessor(client=anthropic_client)
    batcher.add("req-001", messages=[...], model="claude-haiku-4-5-20251001")
    batcher.add("req-002", messages=[...], model="claude-haiku-4-5-20251001")
    batch_id = batcher.submit()             # submit to Anthropic
    results = batcher.poll(batch_id)        # poll for results
"""

import time
import json
import uuid
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional


MAX_BATCH_SIZE = 10_000
POLL_INTERVAL_SECONDS = 30


@dataclass
class BatchRequest:
    custom_id: str
    model: str
    messages: list[dict]
    system: Optional[str | list] = None
    max_tokens: int = 1024
    temperature: float = 0.0
    metadata: dict = field(default_factory=dict)
    added_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class BatchResult:
    custom_id: str
    success: bool
    content: Optional[str] = None
    error: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class BatchProcessor:
    """
    Collects requests and submits them as Anthropic Message Batches.
    Achieves 50% cost savings on non-urgent workloads.

    Example:
        processor = BatchProcessor(client=anthropic_client)

        # Queue requests throughout the day
        processor.add("doc-1", messages=[{"role": "user", "content": "Summarize: ..."}])
        processor.add("doc-2", messages=[{"role": "user", "content": "Summarize: ..."}])

        # Submit batch (returns batch_id)
        batch_id = processor.submit()

        # Poll for results (blocking)
        results = processor.poll_until_complete(batch_id, timeout_seconds=3600)
    """

    def __init__(
        self,
        client=None,                        # anthropic.Anthropic() client
        default_model: str = "claude-haiku-4-5-20251001",
        default_max_tokens: int = 1024,
        auto_submit_at: int = MAX_BATCH_SIZE,   # auto-submit when queue reaches this
        on_submit: Optional[Callable] = None,   # callback when batch is submitted
    ):
        self._client = client
        self.default_model = default_model
        self.default_max_tokens = default_max_tokens
        self.auto_submit_at = auto_submit_at
        self.on_submit = on_submit

        self._queue: list[BatchRequest] = []
        self._lock = threading.Lock()
        self._submitted_batches: dict[str, list[BatchRequest]] = {}

    def add(
        self,
        custom_id: str,
        messages: list[dict],
        model: Optional[str] = None,
        system: Optional[str | list] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Add a request to the batch queue.

        Returns the custom_id for tracking.
        """
        req = BatchRequest(
            custom_id=custom_id or str(uuid.uuid4()),
            model=model or self.default_model,
            messages=messages,
            system=system,
            max_tokens=max_tokens or self.default_max_tokens,
            temperature=temperature,
            metadata=metadata or {},
        )

        with self._lock:
            self._queue.append(req)
            queue_size = len(self._queue)

        if queue_size >= self.auto_submit_at:
            self.submit()

        return req.custom_id

    def queue_size(self) -> int:
        with self._lock:
            return len(self._queue)

    def submit(self) -> Optional[str]:
        """
        Submit the current queue as a batch to Anthropic.
        Returns the batch_id (or None if queue is empty or no client).
        """
        with self._lock:
            if not self._queue:
                return None
            batch = list(self._queue[:MAX_BATCH_SIZE])
            self._queue = self._queue[MAX_BATCH_SIZE:]

        if not self._client:
            # Dry-run mode: return a fake batch ID
            batch_id = f"dry_run_{uuid.uuid4().hex[:8]}"
            self._submitted_batches[batch_id] = batch
            print(f"[BatchProcessor] Dry-run: would submit {len(batch)} requests (batch: {batch_id})")
            return batch_id

        # Build Anthropic batch requests
        requests = []
        for req in batch:
            body: dict[str, Any] = {
                "model": req.model,
                "max_tokens": req.max_tokens,
                "messages": req.messages,
            }
            if req.system:
                body["system"] = req.system
            if req.temperature != 0.0:
                body["temperature"] = req.temperature

            requests.append({
                "custom_id": req.custom_id,
                "params": body,
            })

        try:
            response = self._client.beta.messages.batches.create(requests=requests)
            batch_id = response.id
            self._submitted_batches[batch_id] = batch
            print(f"[BatchProcessor] Submitted {len(batch)} requests → batch: {batch_id}")
            if self.on_submit:
                self.on_submit(batch_id, len(batch))
            return batch_id
        except Exception as e:
            print(f"[BatchProcessor] Submission failed: {e}")
            # Re-queue requests on failure
            with self._lock:
                self._queue = batch + self._queue
            raise

    def poll(self, batch_id: str) -> Optional[dict]:
        """
        Check the status of a submitted batch.

        Returns:
            dict with "status" and optionally "results"
        """
        if not self._client:
            return {"status": "dry_run", "batch_id": batch_id}

        try:
            batch = self._client.beta.messages.batches.retrieve(batch_id)
            status = batch.processing_status

            if status == "ended":
                results = self._collect_results(batch_id)
                return {"status": "complete", "batch_id": batch_id, "results": results}
            else:
                return {
                    "status": status,
                    "batch_id": batch_id,
                    "request_counts": getattr(batch, "request_counts", {}),
                }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def poll_until_complete(
        self,
        batch_id: str,
        timeout_seconds: int = 3600,
        poll_interval: int = POLL_INTERVAL_SECONDS,
        on_progress: Optional[Callable] = None,
    ) -> list[BatchResult]:
        """
        Block until batch is complete or timeout is reached.

        Args:
            batch_id: Batch to monitor
            timeout_seconds: Give up after this many seconds
            poll_interval: How often to poll (seconds)
            on_progress: Optional callback(status_dict) on each poll
        """
        start = time.time()
        while time.time() - start < timeout_seconds:
            status = self.poll(batch_id)
            if on_progress:
                on_progress(status)

            if status.get("status") in ("complete", "dry_run"):
                return status.get("results", [])
            if status.get("status") == "error":
                raise RuntimeError(f"Batch error: {status.get('error')}")

            elapsed = int(time.time() - start)
            print(f"[BatchProcessor] Batch {batch_id} — {status.get('status')} ({elapsed}s elapsed)")
            time.sleep(poll_interval)

        raise TimeoutError(f"Batch {batch_id} did not complete within {timeout_seconds}s")

    def _collect_results(self, batch_id: str) -> list[BatchResult]:
        """Collect and parse batch results from Anthropic."""
        results = []
        try:
            for result in self._client.beta.messages.batches.results(batch_id):
                if result.result.type == "succeeded":
                    msg = result.result.message
                    content = ""
                    for block in msg.content:
                        if hasattr(block, "text"):
                            content += block.text
                    results.append(BatchResult(
                        custom_id=result.custom_id,
                        success=True,
                        content=content,
                        input_tokens=msg.usage.input_tokens,
                        output_tokens=msg.usage.output_tokens,
                    ))
                else:
                    error = result.result.error if hasattr(result.result, "error") else "Unknown error"
                    results.append(BatchResult(
                        custom_id=result.custom_id,
                        success=False,
                        error=str(error),
                    ))
        except Exception as e:
            print(f"[BatchProcessor] Failed to collect results: {e}")
        return results

    def to_jsonl(self, filepath: str):
        """Export current queue to JSONL for manual submission or inspection."""
        with self._lock:
            batch = list(self._queue)

        with open(filepath, "w") as f:
            for req in batch:
                record = {
                    "custom_id": req.custom_id,
                    "params": {
                        "model": req.model,
                        "max_tokens": req.max_tokens,
                        "messages": req.messages,
                    },
                }
                if req.system:
                    record["params"]["system"] = req.system
                f.write(json.dumps(record) + "\n")
        print(f"[BatchProcessor] Exported {len(batch)} requests to {filepath}")
