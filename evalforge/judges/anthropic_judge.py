"""Anthropic Claude-based judge.

Mirrors OpenAIJudge but uses the anthropic SDK. Lazy imports so the package
stays optional.
"""

from __future__ import annotations

import os

from evalforge.judges.base import BaseJudge


class AnthropicJudge(BaseJudge):
    """Judge backed by Anthropic Claude."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-latest",
        n_samples: int = 3,
        temperature: float = 0.7,
        api_key: str | None = None,
    ):
        super().__init__(model=model, n_samples=n_samples, temperature=temperature)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "AnthropicJudge requires the 'anthropic' package. "
                "Install with: pip install anthropic"
            ) from e
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Either pass api_key=... or "
                "set the environment variable."
            )
        self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _call(self, prompt: str, temperature: float) -> str:
        client = self._get_client()
        resp = client.messages.create(
            model=self.model,
            max_tokens=500,
            temperature=temperature,
            system="You are a strict evaluator. Respond with JSON only.",
            messages=[{"role": "user", "content": prompt}],
        )
        # Anthropic returns a list of content blocks
        text_blocks = [b.text for b in resp.content if hasattr(b, "text")]
        return "\n".join(text_blocks)
