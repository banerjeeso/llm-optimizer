"""
Streaming support for OptimizedClient.

All optimizations (caching, routing, prompt optimization) apply identically
to streaming requests. The only difference is the response is a generator
that yields text chunks as they arrive, rather than a single response object.

Usage:
    with client.stream(messages=[...], system="...") as stream:
        for text in stream:
            print(text, end="", flush=True)

    # Access final usage after stream completes
    usage = stream.usage()
"""

from typing import Any, Generator, Optional
from contextlib import contextmanager


class StreamResult:
    """
    Wraps a streaming response to provide a consistent interface
    and capture usage metadata after the stream ends.
    """

    def __init__(self, raw_stream, provider: str):
        self._raw = raw_stream
        self._provider = provider
        self._usage: Optional[dict] = None
        self._finished = False

    def __iter__(self) -> Generator[str, None, None]:
        """Yield text chunks as they arrive."""
        if self._provider == "anthropic":
            yield from self._iter_anthropic()
        elif self._provider == "openai":
            yield from self._iter_openai()
        else:
            raise NotImplementedError(f"Streaming not implemented for provider: {self._provider}")

    def _iter_anthropic(self):
        with self._raw as stream:
            for event in stream:
                # text_delta events carry the actual content
                if hasattr(event, "type"):
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            yield event.delta.text
                    elif event.type == "message_delta":
                        # Capture usage from the final message_delta event
                        if hasattr(event, "usage"):
                            self._usage = {
                                "input_tokens": getattr(event.usage, "input_tokens", 0) or 0,
                                "output_tokens": getattr(event.usage, "output_tokens", 0) or 0,
                                "cache_read_input_tokens": getattr(event.usage, "cache_read_input_tokens", 0) or 0,
                            }

        self._finished = True

    def _iter_openai(self):
        for chunk in self._raw:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    yield delta.content
        self._finished = True

    def usage(self) -> Optional[dict]:
        """
        Returns token usage after stream is fully consumed.
        Returns None if the stream has not been fully consumed yet.
        """
        if not self._finished:
            return None
        return self._usage

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def build_anthropic_stream(client, model_id: str, messages: list, system: Optional[list],
                            max_tokens: int, temperature: float, **kwargs) -> StreamResult:
    """Create an Anthropic streaming request."""
    params: dict[str, Any] = {
        "model": model_id,
        "max_tokens": max_tokens,
        "messages": messages,
        **kwargs,
    }
    if system:
        params["system"] = system
    if temperature > 0:
        params["temperature"] = temperature

    raw = client.messages.stream(**params)
    return StreamResult(raw, provider="anthropic")


def build_openai_stream(client, model_id: str, messages: list, system: Optional[list],
                         max_tokens: int, temperature: float, **kwargs) -> StreamResult:
    """Create an OpenAI streaming request."""
    all_messages = []
    if system:
        sys_text = " ".join(b.get("text", "") for b in system if b.get("type") == "text")
        all_messages.append({"role": "system", "content": sys_text})
    all_messages.extend(messages)

    raw = client.chat.completions.create(
        model=model_id,
        messages=all_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
        **kwargs,
    )
    return StreamResult(raw, provider="openai")
