"""Tests for the runner and end-to-end evaluation."""

import json

from evalforge import AnswerRelevance, Groundedness, run_evaluation
from evalforge.datasets import load_hotpotqa_sample, load_triviaqa_sample


def test_run_evaluation_end_to_end(mock_judge, tmp_path):
    cases = load_triviaqa_sample()
    metrics = [AnswerRelevance(mock_judge), Groundedness(mock_judge)]

    suite = run_evaluation(cases, metrics, suite_name="e2e_test", max_workers=1)
    assert len(suite.results) == len(cases)
    for r in suite.results:
        assert "answer_relevance" in r.scores
        assert "groundedness" in r.scores

    out = tmp_path / "suite.json"
    suite.save_json(out)
    data = json.loads(out.read_text())
    assert data["suite_name"] == "e2e_test"
    assert len(data["results"]) == len(cases)


def test_run_evaluation_parallel(mock_judge):
    cases = load_hotpotqa_sample()
    metrics = [AnswerRelevance(mock_judge)]
    suite = run_evaluation(cases, metrics, suite_name="parallel", max_workers=4)
    assert len(suite.results) == len(cases)
    # Order should be preserved
    assert [r.case_id for r in suite.results] == [c.case_id for c in cases]


def test_suite_summary_table(mock_judge):
    cases = load_triviaqa_sample()
    metrics = [AnswerRelevance(mock_judge)]
    suite = run_evaluation(cases, metrics, suite_name="summary_test", max_workers=1)
    text = suite.summary_table()
    assert "summary_test" in text
    assert "answer_relevance" in text
