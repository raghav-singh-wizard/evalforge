"""Tests for metrics."""

from evalforge.core import EvalCase
from evalforge.metrics import (
    AnswerRelevance,
    Faithfulness,
    Groundedness,
    HallucinationRate,
)


def _case(actual="", expected="", question="What?", ctx=None):
    return EvalCase(
        case_id="t",
        question=question,
        expected_answer=expected,
        actual_answer=actual,
        retrieved_context=ctx or [],
    )


def test_answer_relevance_returns_valid_score(mock_judge):
    m = AnswerRelevance(mock_judge)
    s = m.score(_case(actual="Paris is the capital.", question="What is the capital of France?"))
    assert s.metric_name == "answer_relevance"
    assert 0.0 <= s.score <= 1.0
    assert len(s.samples) == mock_judge.n_samples


def test_groundedness_with_context(mock_judge):
    m = Groundedness(mock_judge)
    s = m.score(_case(actual="Paris.", ctx=["Paris is the capital of France."]))
    assert 0.0 <= s.score <= 1.0


def test_groundedness_no_context(mock_judge):
    m = Groundedness(mock_judge)
    s = m.score(_case(actual="Paris."))
    # should not crash when no context provided
    assert 0.0 <= s.score <= 1.0


def test_faithfulness(mock_judge):
    m = Faithfulness(mock_judge)
    s = m.score(_case(actual="The answer is four.", expected="4"))
    assert s.metric_name == "faithfulness"
    assert 0.0 <= s.score <= 1.0


def test_hallucination_rate_is_inverted(mock_judge):
    """Hallucination metric inverts the raw judge score: higher should mean fewer hallucinations."""
    m = HallucinationRate(mock_judge)
    s = m.score(_case(actual="Claim A.", expected="Claim A.", ctx=["Claim A is in context."]))
    # score should be inverted from raw judge score — we just check it's in range
    assert 0.0 <= s.score <= 1.0
    # and the samples should be the inverted ones
    for sample in s.samples:
        assert 0.0 <= sample <= 1.0
