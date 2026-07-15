"""
AWS Bedrock support for llm-optimizer.

Wraps boto3's bedrock-runtime client to match the same interface
as anthropic.Anthropic(), so all optimizations apply identically.

Supports two auth patterns:

  Option 1 — AWS profile (most common for local dev):
    from llm_optimizer import OptimizedClient, Provider
    from llm_optimizer.bedrock import BedrockClient

    client = OptimizedClient(
        bedrock_client=BedrockClient(profile_name="ai-core", region="us-east-1"),
        preferred_provider=Provider.BEDROCK,
    )

  Option 2 — Env vars (CI/CD pipelines, Lambda, EC2 with IAM role):
    # Set: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
    # or attach an IAM role — boto3 picks it up automatically

    client = OptimizedClient(
        bedrock_client=BedrockClient(),   # zero config — reads env/role
        preferred_provider=Provider.BEDROCK,
    )

Key differences from direct Anthropic API:
  - Auth via AWS IAM, not API keys
  - Prompt caching uses Bedrock cachePoint format (not cache_control) — translated automatically
  - Automatic caching NOT supported on Bedrock — explicit checkpoints injected instead
  - Min cacheable tokens: 4,096 (vs 1,024 on direct API) for Claude 4.5 models
  - Model IDs use Bedrock format: anthropic.claude-haiku-4-5-20251001-v1:0
  - Batch via Bedrock Batch Inference (not Anthropic Batch API)
  - TTL preserved correctly (LiteLLM has a bug that strips TTL on Bedrock — this library does not)

Install:
  pip install llm-optimizer[bedrock]
"""

import json
import warnings
from dataclasses import dataclass
from typing import Any, Generator, Optional


@dataclass
class BedrockResponse:
    """
    Normalizes Bedrock response to match Anthropic SDK response shape.
    Ensures tracker, streaming, and all other components work identically.
    """
    content: list
    usage: "BedrockUsage"
    model: str
    stop_reason: str

    def __str__(self):
        return self.content[0].text if self.content else ""


@dataclass
class BedrockUsage:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0  # Always 0 — Bedrock does not support caching


@dataclass
class BedrockContentBlock:
    text: str
    type: str = "text"


class BedrockClient:
    """
    Drop-in Bedrock client for OptimizedClient.

    Auth priority (boto3 standard credential chain):
      1. profile_name argument  → ~/.aws/credentials [profile]
      2. session argument       → pre-built boto3.Session
      3. Environment variables  → AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
      4. IAM role               → auto-detected on EC2 / Lambda / ECS

    Args:
        region:        AWS region (default: us-east-1)
        profile_name:  AWS profile from ~/.aws/config (e.g. "ai-core")
        session:       Pre-configured boto3.Session — overrides profile_name
        discount_pct:  Your negotiated AWS discount (e.g. 16.0).
                       Applied to cost estimates only — does not affect API calls.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        profile_name: Optional[str] = None,
        session=None,
        discount_pct: float = 0.0,
        enable_caching: bool = True,
        cache_ttl: str = "5m",
    ):
        """
        Args:
            region:         AWS region (default: us-east-1)
            profile_name:   AWS profile from ~/.aws/config (e.g. "ai-core")
            session:        Pre-configured boto3.Session — overrides profile_name
            discount_pct:   Negotiated AWS discount % for cost estimates (e.g. 16.0)
            enable_caching: Enable Bedrock prompt caching (default: True)
                            Translates Anthropic cache_control → Bedrock cachePoint format
                            Injects optimal checkpoints when none present
            cache_ttl:      Cache TTL — "5m" (default) or "1h"
                            1h supported on Claude Haiku/Sonnet/Opus 4.5 only
        """
        self.region = region
        self.discount_pct = discount_pct
        self._profile = profile_name
        self._session = session
        self._runtime = None
        self._boto3 = None

        # Import here to avoid circular imports
        from .bedrock_cache import BedrockCacheManager
        self.cache_manager = BedrockCacheManager(
            enabled=enable_caching,
            ttl=cache_ttl,
        )

    # ── Credential chain ───────────────────────────────────────────────────────

    def _get_runtime(self):
        """Lazy-initialize boto3 bedrock-runtime client."""
        if self._runtime:
            return self._runtime

        try:
            import boto3
            self._boto3 = boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for AWS Bedrock support.\n"
                "Install with: pip install llm-optimizer[bedrock]"
            )

        # Credential chain: explicit session > profile > env/role
        if self._session:
            resolved_session = self._session
        elif self._profile:
            resolved_session = boto3.Session(profile_name=self._profile, region_name=self.region)
        else:
            # Falls through to env vars, then IAM role
            resolved_session = boto3.Session(region_name=self.region)

        self._runtime = resolved_session.client(
            service_name="bedrock-runtime",
            region_name=self.region,
        )
        return self._runtime

    # ── messages interface (matches anthropic.Anthropic().messages) ────────────

    class _Messages:
        def __init__(self, bedrock_client: "BedrockClient"):
            self._client = bedrock_client

        def create(self, model, max_tokens, messages, system=None, **kwargs) -> BedrockResponse:
            return self._client._invoke(model, max_tokens, messages, system, **kwargs)

        def stream(self, model, max_tokens, messages, system=None, **kwargs):
            return _BedrockStreamContext(self._client, model, max_tokens, messages, system, **kwargs)

    @property
    def messages(self):
        return self._Messages(self)

    # ── Effective cost (with discount) ────────────────────────────────────────

    def effective_cost_multiplier(self) -> float:
        """Returns 1.0 - (discount_pct / 100). Applied to pricing estimates."""
        return 1.0 - (self.discount_pct / 100.0)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_body(
        self,
        model: str,
        max_tokens: int,
        messages: list[dict],
        system: Optional[Any],
        temperature: float = 0.0,
        **kwargs,
    ) -> dict:
        """
        Build Bedrock request body.

        Translates Anthropic cache_control blocks → Bedrock cachePoint blocks.
        Injects optimal cachePoint checkpoints when none present.

        Note: Bedrock uses cachePoint (not cache_control).
        Bedrock does NOT support automatic caching — we inject explicit checkpoints.
        TTL is preserved correctly (LiteLLM strips it — this library does not).
        """
        # Translate/inject Bedrock cachePoint blocks
        prepared_messages, prepared_system = self.cache_manager.prepare(
            messages=messages,
            system=system,
        )

        # Build system text from translated blocks
        system_text = ""
        system_blocks_for_body = None
        if prepared_system:
            # Check if any block is a cachePoint — if so, keep as blocks list
            has_cache_points = any(
                isinstance(b, dict) and "cachePoint" in b
                for b in prepared_system
            )
            if has_cache_points:
                system_blocks_for_body = prepared_system
            else:
                system_text = " ".join(
                    b.get("text", "") for b in prepared_system
                    if isinstance(b, dict) and b.get("type") == "text"
                )

        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": prepared_messages,
        }

        if system_blocks_for_body:
            body["system"] = system_blocks_for_body
        elif system_text:
            body["system"] = system_text

        if temperature > 0:
            body["temperature"] = temperature

        return body

    def _invoke(self, model, max_tokens, messages, system=None, **kwargs) -> BedrockResponse:
        runtime = self._get_runtime()
        body = self._build_body(model, max_tokens, messages, system, **kwargs)

        response = runtime.invoke_model(
            modelId=model,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())

        content_blocks = [
            BedrockContentBlock(text=block["text"])
            for block in result.get("content", [])
            if block.get("type") == "text"
        ]

        usage_data = result.get("usage", {})
        return BedrockResponse(
            content=content_blocks,
            usage=BedrockUsage(
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
            ),
            model=model,
            stop_reason=result.get("stop_reason", "end_turn"),
        )

    @staticmethod
    def _strip_cache_control(messages: list[dict]) -> list[dict]:
        """
        Fallback: strip cache_control blocks when caching is disabled.
        When caching is enabled, bedrock_cache.py translates cache_control
        to Bedrock's cachePoint format instead of stripping it.
        """
        clean = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                clean_blocks = [
                    {k: v for k, v in block.items() if k != "cache_control"}
                    if isinstance(block, dict) else block
                    for block in content
                ]
                clean.append({**msg, "content": clean_blocks})
            else:
                clean.append(msg)
        return clean


class _BedrockStreamContext:
    """Context manager for Bedrock streaming. Matches Anthropic stream interface."""

    def __init__(self, bedrock_client, model, max_tokens, messages, system, **kwargs):
        self._client = bedrock_client
        self._model = model
        self._max_tokens = max_tokens
        self._messages = messages
        self._system = system
        self._kwargs = kwargs
        self.usage = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self) -> Generator[str, None, None]:
        runtime = self._client._get_runtime()
        body = self._client._build_body(
            self._model, self._max_tokens, self._messages, self._system, **self._kwargs
        )

        response = runtime.invoke_model_with_response_stream(
            modelId=self._model,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        input_tokens = output_tokens = 0

        for event in response.get("body", []):
            chunk = event.get("chunk", {})
            if not chunk:
                continue
            data = json.loads(chunk.get("bytes", b"{}"))
            etype = data.get("type", "")

            if etype == "content_block_delta":
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta":
                    yield delta.get("text", "")
            elif etype == "message_start":
                input_tokens = data.get("message", {}).get("usage", {}).get("input_tokens", 0)
            elif etype == "message_delta":
                output_tokens = data.get("usage", {}).get("output_tokens", 0)

        self.usage = BedrockUsage(input_tokens=input_tokens, output_tokens=output_tokens)


# ── Convenience factory ────────────────────────────────────────────────────────

def make_bedrock_client(
    profile_name: Optional[str] = None,
    region: str = "us-east-1",
    discount_pct: float = 0.0,
    enable_caching: bool = True,
    cache_ttl: str = "5m",
) -> BedrockClient:
    """
    Convenience factory for BedrockClient.

    Prompt caching is enabled by default. The library automatically:
      - Translates Anthropic cache_control → Bedrock cachePoint format
      - Injects optimal checkpoints when none are present
      - Preserves TTL correctly (1h supported for Claude 4.5 models)

    Usage:
        # Profile-based (most team devs) — caching on by default
        bedrock = make_bedrock_client(profile_name="ai-core", discount_pct=16.0)

        # With 1-hour TTL (Claude 4.5 models only)
        bedrock = make_bedrock_client(profile_name="ai-core", cache_ttl="1h")

        # Env-var / IAM role (CI/CD, Lambda)
        bedrock = make_bedrock_client()

        # Disable caching explicitly
        bedrock = make_bedrock_client(enable_caching=False)
    """
    return BedrockClient(
        region=region,
        profile_name=profile_name,
        discount_pct=discount_pct,
        enable_caching=enable_caching,
        cache_ttl=cache_ttl,
    )
