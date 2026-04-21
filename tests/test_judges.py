"""Tests for judges."""

from evalforge.judges.base import BaseJudge
from evalforge.judges.mock import MockJudge


def test_mock_judge_deterministic_at_zero_temp():
    j = MockJudge(n_samples=1, temperature=0.0)
    r1 = j.grade("question: foo\nanswer: bar\ncontext: foo bar")
    r2 = j.grade("question: foo\nanswer: bar\ncontext: foo bar")
    assert abs(r1.score - r2.score) < 1e-9


def test_mock_judge_self_consistency_returns_samples():
    j = MockJudge(n_samples=3, temperature=0.7)
    response, samples = j.grade_with_samples("question: x\nanswer: y")
    assert len(samples) == 3
    # score should be mean of samples
    mean = sum(samples) / len(samples)
    assert abs(response.score - mean) < 1e-9


def test_parse_response_json():
    raw = 'Some preamble... {"score": 0.85, "reasoning": "good match"} trailing'
    parsed = BaseJudge._parse_response(raw)
    assert parsed.score == 0.85
    assert "good match" in parsed.reasoning


def test_parse_response_regex_fallback():
    raw = "After careful analysis the score: 0.72 seems right."
    parsed = BaseJudge._parse_response(raw)
    assert abs(parsed.score - 0.72) < 1e-9


def test_parse_response_clamps():
    raw = '{"score": 1.5, "reasoning": "over"}'
    parsed = BaseJudge._parse_response(raw)
    assert parsed.score == 1.0

    raw = '{"score": -0.3, "reasoning": "under"}'
    parsed = BaseJudge._parse_response(raw)
    assert parsed.score == 0.0


def test_parse_response_ten_scale():
    raw = "score: 8"
    parsed = BaseJudge._parse_response(raw)
    # 8 > 1.0 => interpreted as 0-10 => 0.8
    assert abs(parsed.score - 0.8) < 1e-9


def test_parse_response_unparseable_returns_neutral():
    parsed = BaseJudge._parse_response("no useful content")
    assert parsed.score == 0.5
