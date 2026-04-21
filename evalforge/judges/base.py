"""Base judge interface with self-consistency sampling support."""

from __future__ import annotations

import json
import re
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class JudgeResponse:
    """What a judge returns for a single call."""

    score: float  # 0.0 to 1.0
    reasoning: str
    raw_response: str = ""


class BaseJudge(ABC):
    """Abstract base for LLM judges.

    Self-consistency sampling runs the judge N times and aggregates. This
    stabilizes scores against the non-determinism inherent to LLM outputs.
    """

    def __init__(self, model: str, n_samples: int = 3, temperature: float = 0.7):
        self.model = model
        self.n_samples = n_samples
        self.temperature = temperature

    @abstractmethod
    def _call(self, prompt: str, temperature: float) -> str:
        """Single call to the underlying LLM. Returns raw text."""
        ...

    def grade(self, prompt: str) -> JudgeResponse:
        """Grade with self-consistency: sample N times, aggregate."""
        samples: list[float] = []
        reasonings: list[str] = []
        raw = ""

        for i in range(self.n_samples):
            # First sample at low temp for a stable baseline; subsequent samples
            # at the configured temperature to capture variance.
            temp = 0.0 if i == 0 else self.temperature
            raw = self._call(prompt, temperature=temp)
            parsed = self._parse_response(raw)
            samples.append(parsed.score)
            reasonings.append(parsed.reasoning)

        # Aggregate: mean score, first reasoning (usually the most stable)
        mean_score = statistics.mean(samples)
        return JudgeResponse(
            score=mean_score,
            reasoning=reasonings[0],
            raw_response=raw,
        )

    def grade_with_samples(self, prompt: str) -> tuple[JudgeResponse, list[float]]:
        """Same as grade() but also returns the per-sample scores."""
        samples: list[float] = []
        reasonings: list[str] = []

        for i in range(self.n_samples):
            temp = 0.0 if i == 0 else self.temperature
            raw = self._call(prompt, temperature=temp)
            parsed = self._parse_response(raw)
            samples.append(parsed.score)
            reasonings.append(parsed.reasoning)

        mean_score = statistics.mean(samples)
        response = JudgeResponse(
            score=mean_score,
            reasoning=reasonings[0],
        )
        return response, samples

    @staticmethod
    def _parse_response(raw: str) -> JudgeResponse:
        """Extract score and reasoning from a judge response.

        Expects a JSON block like {"score": 0.8, "reasoning": "..."} anywhere in the text.
        Falls back to regex extraction if JSON parsing fails.
        """
        # Try JSON first
        json_match = re.search(r"\{[^{}]*\"score\"[^{}]*\}", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                score = float(data.get("score", 0.0))
                score = max(0.0, min(1.0, score))  # clamp
                return JudgeResponse(
                    score=score,
                    reasoning=str(data.get("reasoning", "")),
                    raw_response=raw,
                )
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: regex for "score: X.Y" or "0.X"
        score_match = re.search(r"score[:\s]+(\d*\.?\d+)", raw, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                if score > 1.0:  # likely on a 0-10 scale
                    score = score / 10.0
                score = max(0.0, min(1.0, score))
                return JudgeResponse(score=score, reasoning=raw[:200], raw_response=raw)
            except ValueError:
                pass

        # Give up — return neutral
        return JudgeResponse(score=0.5, reasoning=f"Could not parse: {raw[:100]}", raw_response=raw)
