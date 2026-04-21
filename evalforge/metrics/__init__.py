"""Evaluation metrics.

All metrics implement:
    score(case: EvalCase) -> MetricScore

Metrics take a judge at construction time.
"""

from evalforge.metrics.answer_relevance import AnswerRelevance
from evalforge.metrics.base import BaseMetric
from evalforge.metrics.faithfulness import Faithfulness
from evalforge.metrics.groundedness import Groundedness
from evalforge.metrics.hallucination import HallucinationRate

__all__ = [
    "BaseMetric",
    "AnswerRelevance",
    "Faithfulness",
    "Groundedness",
    "HallucinationRate",
]
