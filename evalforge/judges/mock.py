"""Mock judge for testing and CI without API calls.

The mock uses simple heuristics (token overlap, length ratio) to produce
deterministic-ish scores. Not a serious evaluator — only for testing the harness itself.
"""

from __future__ import annotations

import hashlib
import random
import re

from evalforge.judges.base import BaseJudge


class MockJudge(BaseJudge):
    """A deterministic, no-API-call judge for testing.

    Scores are derived from:
      - token overlap between answer and context/expected
      - a pseudo-random jitter seeded by prompt hash (for self-consistency variance)
    """

    def __init__(self, model: str = "mock", n_samples: int = 3, temperature: float = 0.7):
        super().__init__(model=model, n_samples=n_samples, temperature=temperature)

    def _call(self, prompt: str, temperature: float) -> str:
        # Deterministic base score from prompt hash
        h = hashlib.sha256(prompt.encode()).hexdigest()
        base = int(h[:4], 16) / 0xFFFF  # 0..1

        # Parse a simple signal: count of word-overlap hints in prompt
        overlap_signal = self._token_overlap_signal(prompt)

        # Combine: 60% overlap, 40% base hash
        score = 0.6 * overlap_signal + 0.4 * base

        # Add jitter proportional to temperature
        rng = random.Random(int(h[:8], 16) + int(temperature * 1000))
        jitter = rng.uniform(-0.05, 0.05) * temperature
        score = max(0.0, min(1.0, score + jitter))

        reasoning = f"Mock judgment based on {overlap_signal:.2f} overlap signal."
        return f'{{"score": {score:.3f}, "reasoning": "{reasoning}"}}'

    @staticmethod
    def _token_overlap_signal(prompt: str) -> float:
        """Crude proxy: if 'answer' and 'context' sections share many tokens, return high."""
        # Split on common section markers
        sections = re.split(r"(?i)(answer|context|expected|question)[:\s]", prompt)
        if len(sections) < 3:
            return 0.5

        tokens_lists = []
        for sec in sections:
            words = set(re.findall(r"\w+", sec.lower()))
            if len(words) >= 3:
                tokens_lists.append(words)

        if len(tokens_lists) < 2:
            return 0.5

        # Jaccard between first two substantive sections
        a, b = tokens_lists[0], tokens_lists[1]
        if not a or not b:
            return 0.5
        jaccard = len(a & b) / len(a | b)
        return min(1.0, jaccard * 2.5)  # scale up a bit — pure jaccard is conservative
