"""Tests for core data types."""

import json

from evalforge.core import EvalCase, EvalResult, EvalSuite, MetricScore


def test_evalcase_roundtrip():
    case = EvalCase(
        case_id="c1",
        question="What is 2+2?",
        expected_answer="4",
        actual_answer="Four.",
        retrieved_context=["math basics"],
    )
    d = case.to_dict()
    assert d["case_id"] == "c1"
    assert d["retrieved_context"] == ["math basics"]


def test_metricscore_std_empty():
    s = MetricScore(metric_name="m", score=0.5)
    assert s.std == 0.0


def test_metricscore_std_nontrivial():
    s = MetricScore(metric_name="m", score=0.5, samples=[0.4, 0.5, 0.6])
    assert s.std > 0.0


def test_evalresult_aggregate():
    r = EvalResult(case_id="c1")
    r.scores["a"] = MetricScore("a", 0.8)
    r.scores["b"] = MetricScore("b", 0.6)
    assert abs(r.aggregate() - 0.7) < 1e-9


def test_evalresult_aggregate_empty():
    r = EvalResult(case_id="c1")
    assert r.aggregate() == 0.0


def test_evalsuite_metric_means():
    suite = EvalSuite(name="s")
    r1 = EvalResult(case_id="c1")
    r1.scores["a"] = MetricScore("a", 1.0)
    r1.scores["b"] = MetricScore("b", 0.4)
    r2 = EvalResult(case_id="c2")
    r2.scores["a"] = MetricScore("a", 0.5)
    r2.scores["b"] = MetricScore("b", 0.8)
    suite.add(r1)
    suite.add(r2)

    means = suite.metric_means()
    assert abs(means["a"] - 0.75) < 1e-9
    assert abs(means["b"] - 0.6) < 1e-9
    assert abs(suite.overall_score() - 0.675) < 1e-9


def test_evalsuite_pass_rate():
    suite = EvalSuite(name="s")
    for i, agg_scores in enumerate([(0.9, 0.8), (0.3, 0.4), (0.7, 0.7)]):
        r = EvalResult(case_id=f"c{i}")
        r.scores["a"] = MetricScore("a", agg_scores[0])
        r.scores["b"] = MetricScore("b", agg_scores[1])
        suite.add(r)
    # Aggregates: 0.85, 0.35, 0.7 -> 2 of 3 pass at threshold 0.7
    assert abs(suite.pass_rate(threshold=0.7) - 2 / 3) < 1e-9


def test_evalsuite_save_and_load_json(tmp_path):
    suite = EvalSuite(name="test_suite", metadata={"model": "mock"})
    r = EvalResult(case_id="c1")
    r.scores["a"] = MetricScore("a", 0.9, reasoning="looks good")
    suite.add(r)

    path = tmp_path / "out.json"
    suite.save_json(path)
    assert path.exists()

    loaded = json.loads(path.read_text())
    assert loaded["suite_name"] == "test_suite"
    assert loaded["metadata"]["model"] == "mock"
    assert loaded["metric_means"]["a"] == 0.9
    assert len(loaded["results"]) == 1
