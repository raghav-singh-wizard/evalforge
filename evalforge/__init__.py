"""EvalForge — Open-Source LLM Evaluation Harness.

A pytest-integrated framework for evaluating non-deterministic LLM pipelines.
Measures groundedness, faithfulness, answer relevance, and hallucination rate
using LLM-as-a-judge with self-consistency sampling.
"""

from evalforge.core import EvalCase, EvalResult, EvalSuite
from evalforge.metrics.answer_relevance import AnswerRelevance
from evalforge.metrics.faithfulness import Faithfulness
from evalforge.metrics.groundedness import Groundedness
from evalforge.metrics.hallucination import HallucinationRate
from evalforge.runner import run_evaluation

__version__ = "0.1.0"

__all__ = [
    "EvalCase",
    "EvalResult",
    "EvalSuite",
    "AnswerRelevance",
    "Faithfulness",
    "Groundedness",
    "HallucinationRate",
    "run_evaluation",
]
