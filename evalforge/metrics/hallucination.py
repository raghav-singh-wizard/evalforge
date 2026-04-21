"""Hallucination Rate.

Does the answer introduce claims NOT supported by context and NOT in the
expected answer? Higher score here = MORE hallucination, so we invert at the
end so that higher is better (1.0 = no hallucinations).
"""

from __future__ import annotations

from evalforge.core import EvalCase, MetricScore
from evalforge.metrics.base import BaseMetric

PROMPT_TEMPLATE = """You are detecting HALLUCINATIONS in an ANSWER.
A hallucination is a claim that is NOT in the context AND NOT in the expected answer
— the model invented it.

CONTEXT:
{context}

EXPECTED ANSWER:
{expected}

ACTUAL ANSWER:
{actual}

Score the HALLUCINATION RATE from 0.0 to 1.0:
  - 0.0: no hallucinations — every claim is supported by context or matches expected
  - 0.5: some claims appear invented
  - 1.0: most or all claims are invented

Respond with JSON only:
{{"score": <float>, "reasoning": "<one sentence>"}}"""


class HallucinationRate(BaseMetric):
    name = "hallucination_rate"

    def _build_prompt(self, case: EvalCase) -> str:
        context_str = (
            "\n\n---\n\n".join(case.retrieved_context)
            if case.retrieved_context
            else "(no context provided)"
        )
        return PROMPT_TEMPLATE.format(
            context=context_str,
            expected=case.expected_answer,
            actual=case.actual_answer,
        )

    def score(self, case: EvalCase) -> MetricScore:
        # Inherit the judge call, then invert so higher = better
        raw = super().score(case)
        inverted = MetricScore(
            metric_name=self.name,
            score=1.0 - raw.score,
            reasoning=raw.reasoning,
            samples=[1.0 - s for s in raw.samples],
        )
        return inverted
