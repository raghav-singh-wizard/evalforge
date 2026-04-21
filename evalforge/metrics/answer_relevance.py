"""Answer Relevance.

Does the actual answer address the question? Ignores correctness — a wrong
answer can still be relevant (it's about the right thing) or irrelevant
(off-topic entirely).
"""

from __future__ import annotations

from evalforge.core import EvalCase
from evalforge.metrics.base import BaseMetric

PROMPT_TEMPLATE = """You are evaluating whether an ANSWER addresses a given QUESTION.
This is about topical relevance, not correctness.

QUESTION:
{question}

ANSWER:
{answer}

Score from 0.0 to 1.0:
  - 1.0: answer directly addresses the question
  - 0.5: answer is partially on-topic
  - 0.0: answer is completely off-topic or empty

Respond with JSON only:
{{"score": <float>, "reasoning": "<one sentence>"}}"""


class AnswerRelevance(BaseMetric):
    name = "answer_relevance"

    def _build_prompt(self, case: EvalCase) -> str:
        return PROMPT_TEMPLATE.format(
            question=case.question,
            answer=case.actual_answer,
        )
