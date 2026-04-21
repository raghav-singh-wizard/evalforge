"""OpenAI GPT-based judge.

Uses the openai SDK if available. If the package or API key is missing,
raises a helpful error at grade-time (not import-time) so the rest of
EvalForge stays usable with MockJudge.
"""

from __future__ import annotations

import os

from evalforge.judges.base import BaseJudge


class OpenAIJudge(BaseJudge):
    """Judge backed by an OpenAI chat model (e.g. gpt-4o-mini, gpt-4)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        n_samples: int = 3,
        temperature: float = 0.7,
        api_key: str | None = None,
    ):
        super().__init__(model=model, n_samples=n_samples, temperature=temperature)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAIJudge requires the 'openai' package. "
                "Install with: pip install openai"
            ) from e
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Either pass api_key=... or "
                "set the environment variable."
            )
        self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _call(self, prompt: str, temperature: float) -> str:
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a strict evaluator. Respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=500,
        )
        return resp.choices[0].message.content or ""
