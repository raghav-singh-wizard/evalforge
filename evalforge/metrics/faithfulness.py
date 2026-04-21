"""Faithfulness.

Does the answer match the expected (ground-truth) answer? This is a
semantic-equivalence check — the phrasing can differ, but the meaning should match.
Distinct from groundedness, which checks against context rather than gold answer.
"""

from __future__ import annotations

from evalforge.core import EvalCase
from evalforge.metrics.base import BaseMetric

PROMPT_TEMPLATE = """You are comparing an ACTUAL ANSWER against an EXPECTED ANSWER.
Judge semantic equivalence, not exact wording.

QUESTION:
{question}

EXPECTED ANSWER:
{expected}

ACTUAL ANSWER:
{actual}

Score from 0.0 to 1.0:
  - 1.0: actual answer is semantically equivalent to expected
  - 0.7: actual captures the main point but misses some detail
  - 0.4: actual partially matches (maybe correct direction but missing key facts)
  - 0.0: actual contradicts or is completely different from expected

Respond with JSON only:
{{"score": <float>, "reasoning": "<one sentence>"}}"""


class Faithfulness(BaseMetric):
    name = "faithfulness"

    def _build_prompt(self, case: EvalCase) -> str:
        return PROMPT_TEMPLATE.format(
            question=case.question,
            expected=case.expected_answer,
            actual=case.actual_answer,
        )
