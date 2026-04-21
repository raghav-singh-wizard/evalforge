"""Base metric class."""

from __future__ import annotations

from abc import ABC, abstractmethod

from evalforge.core import EvalCase, MetricScore
from evalforge.judges.base import BaseJudge


class BaseMetric(ABC):
    """All metrics subclass this."""

    name: str = "base"

    def __init__(self, judge: BaseJudge):
        self.judge = judge

    @abstractmethod
    def _build_prompt(self, case: EvalCase) -> str:
        ...

    def score(self, case: EvalCase) -> MetricScore:
        prompt = self._build_prompt(case)
        response, samples = self.judge.grade_with_samples(prompt)
        return MetricScore(
            metric_name=self.name,
            score=response.score,
            reasoning=response.reasoning,
            samples=samples,
        )
