"""Groundedness.

Are the claims in the answer supported by the retrieved context? This is the
classic RAG quality metric — if context has the info and the answer uses it,
groundedness is high.
"""

from __future__ import annotations

from evalforge.core import EvalCase
from evalforge.metrics.base import BaseMetric

PROMPT_TEMPLATE = """You are evaluating whether an ANSWER is grounded in the provided CONTEXT.
Groundedness means every factual claim in the answer can be traced to the context.

CONTEXT:
{context}

ANSWER:
{answer}

Score from 0.0 to 1.0:
  - 1.0: every claim in the answer is supported by the context
  - 0.5: some claims are supported, others are not
  - 0.0: no claims in the answer are supported (or answer contradicts context)

Respond with JSON only:
{{"score": <float>, "reasoning": "<one sentence>"}}"""


class Groundedness(BaseMetric):
    name = "groundedness"

    def _build_prompt(self, case: EvalCase) -> str:
        if not case.retrieved_context:
            context_str = "(no context provided)"
        else:
            context_str = "\n\n---\n\n".join(case.retrieved_context)
        return PROMPT_TEMPLATE.format(
            context=context_str,
            answer=case.actual_answer,
        )
