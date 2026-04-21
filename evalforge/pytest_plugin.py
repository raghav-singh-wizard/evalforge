"""pytest integration for EvalForge.

Lets you write assertions like:

    def test_my_rag(evalforge_suite, mock_judge):
        from evalforge import AnswerRelevance, Groundedness, run_evaluation
        from evalforge.datasets import load_hotpotqa_sample

        metrics = [AnswerRelevance(mock_judge), Groundedness(mock_judge)]
        suite = run_evaluation(load_hotpotqa_sample(), metrics, suite_name="hotpot")
        assert suite.overall_score() >= 0.5
        assert suite.pass_rate(threshold=0.5) >= 0.8

The `mock_judge` and `evalforge_suite` fixtures are provided by this plugin.
"""

from __future__ import annotations

import pytest

from evalforge.judges.mock import MockJudge


@pytest.fixture
def mock_judge() -> MockJudge:
    """Deterministic judge for tests that don't need real LLM calls."""
    return MockJudge(n_samples=2, temperature=0.5)


@pytest.fixture
def evalforge_suite(tmp_path):
    """A fresh temp dir for writing suite JSON during a test."""
    return tmp_path / "suite.json"
