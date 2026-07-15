"""
Streaming Analytics Hook — emits real-time cost events as tokens stream.

Allows web UIs to show the cost accumulating while the model is streaming.
Supports callback-based hooks and SSE (Server-Sent Events) format.

Usage — callback hook:
    def on_token(event):
        print(f"Token #{event['token_count']} | Est. cost: ${event['estimated_cost']:.6f}")

    with client.stream(
        messages=[...],
        on_token_event=on_token,
    ) as stream:
        for chunk in stream:
            print(chunk, end="", flush=True)

Usage — SSE endpoint (FastAPI example):
    from llm_optimizer.analytics import StreamingAnalyticsCollector
    from sse_starlette.sse import EventSourceResponse

    @app.get("/stream")
    async def stream_endpoint():
        collector = StreamingAnalyticsCollector(model_key="claude-sonnet-4")

        async def event_generator():
            with client.stream(messages=[...]) as stream:
                for chunk in stream:
                    collector.record_chunk(chunk)
                    yield {"data": json.dumps(collector.current_stats())}

        return EventSourceResponse(event_generator())
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .tracker import estimate_tokens
from .models import MODELS


@dataclass
class TokenEvent:
    """Emitted for each streamed chunk."""
    chunk_text: str
    token_count: int            # cumulative tokens so far
    elapsed_ms: float           # time since stream start
    tokens_per_second: float    # rolling rate
    estimated_cost: float       # cumulative cost estimate
    model_key: str


@dataclass
class StreamAnalytics:
    """Final analytics after stream completes."""
    model_key: str
    total_chunks: int
    total_chars: int
    estimated_output_tokens: int
    elapsed_ms: float
    tokens_per_second: float
    estimated_cost: float
    cost_per_second: float


class StreamingAnalyticsCollector:
    """
    Collects real-time analytics during a streaming response.

    Attach to any streaming response to get per-chunk cost estimates,
    token rates, and latency metrics as the stream progresses.

    Args:
        model_key:        Model key from MODELS registry
        input_tokens:     Input tokens (from request, for cost calculation)
        on_token_event:   Callback called with each TokenEvent
        emit_every_n:     Emit event every N chunks (default: 1)
    """

    def __init__(
        self,
        model_key: str = "claude-sonnet-4",
        input_tokens: int = 0,
        on_token_event: Optional[Callable[[TokenEvent], None]] = None,
        emit_every_n: int = 1,
    ):
        self.model_key = model_key
        self.input_tokens = input_tokens
        self.on_token_event = on_token_event
        self.emit_every_n = emit_every_n

        self._model = MODELS.get(model_key)
        self._start_time = time.time()
        self._chunks: list[str] = []
        self._chunk_count = 0
        self._total_chars = 0
        self._lock = threading.Lock()

    def record_chunk(self, chunk_text: str) -> Optional[TokenEvent]:
        """
        Record a streamed chunk and optionally emit a TokenEvent.

        Returns the TokenEvent if emitted, None otherwise.
        """
        with self._lock:
            self._chunks.append(chunk_text)
            self._chunk_count += 1
            self._total_chars += len(chunk_text)

            if self._chunk_count % self.emit_every_n != 0:
                return None

            event = self._build_event(chunk_text)

        if self.on_token_event and event:
            try:
                self.on_token_event(event)
            except Exception:
                pass  # never let analytics break the stream

        return event

    def current_stats(self) -> dict:
        """Return current streaming stats as a JSON-serializable dict."""
        with self._lock:
            elapsed = (time.time() - self._start_time) * 1000
            full_text = "".join(self._chunks)
            est_tokens = estimate_tokens(full_text)
            tps = est_tokens / (elapsed / 1000) if elapsed > 0 else 0
            cost = self._estimate_cost(est_tokens)
            return {
                "chunk_count": self._chunk_count,
                "estimated_output_tokens": est_tokens,
                "elapsed_ms": round(elapsed, 1),
                "tokens_per_second": round(tps, 1),
                "estimated_cost_usd": round(cost, 8),
                "model": self.model_key,
            }

    def finalize(self) -> StreamAnalytics:
        """Return final analytics after stream completes."""
        with self._lock:
            elapsed = (time.time() - self._start_time) * 1000
            full_text = "".join(self._chunks)
            est_tokens = estimate_tokens(full_text)
            tps = est_tokens / (elapsed / 1000) if elapsed > 0 else 0
            cost = self._estimate_cost(est_tokens)

        return StreamAnalytics(
            model_key=self.model_key,
            total_chunks=self._chunk_count,
            total_chars=self._total_chars,
            estimated_output_tokens=est_tokens,
            elapsed_ms=round(elapsed, 1),
            tokens_per_second=round(tps, 1),
            estimated_cost=round(cost, 8),
            cost_per_second=round(cost / (elapsed / 1000) if elapsed > 0 else 0, 8),
        )

    def _build_event(self, chunk_text: str) -> TokenEvent:
        elapsed = (time.time() - self._start_time) * 1000
        full_text = "".join(self._chunks)
        est_tokens = estimate_tokens(full_text)
        tps = est_tokens / (elapsed / 1000) if elapsed > 0 else 0
        cost = self._estimate_cost(est_tokens)

        return TokenEvent(
            chunk_text=chunk_text,
            token_count=est_tokens,
            elapsed_ms=round(elapsed, 1),
            tokens_per_second=round(tps, 1),
            estimated_cost=round(cost, 8),
            model_key=self.model_key,
        )

    def _estimate_cost(self, output_tokens: int) -> float:
        if self._model:
            return self._model.estimate_cost(self.input_tokens, output_tokens)
        return 0.0


def wrap_stream_with_analytics(
    stream_iter,
    model_key: str,
    input_tokens: int = 0,
    on_token_event: Optional[Callable] = None,
    emit_every_n: int = 5,
) -> tuple:
    """
    Wrap a streaming iterator with analytics collection.

    Returns:
        (analytics_generator, collector)
        Iterate the generator; access collector.finalize() when done.

    Usage:
        gen, collector = wrap_stream_with_analytics(stream, "claude-sonnet-4")
        for chunk in gen:
            print(chunk, end="")
        final = collector.finalize()
        print(f"Cost: ${final.estimated_cost:.6f}")
    """
    collector = StreamingAnalyticsCollector(
        model_key=model_key,
        input_tokens=input_tokens,
        on_token_event=on_token_event,
        emit_every_n=emit_every_n,
    )

    def _generator():
        for chunk in stream_iter:
            collector.record_chunk(chunk)
            yield chunk

    return _generator(), collector
