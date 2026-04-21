"""Tests for the regression gate."""

import json

from evalforge.regression_gate import compare_suites, main


def _write_suite(path, metric_means):
    payload = {
        "suite_name": "s",
        "metric_means": metric_means,
        "results": [],
    }
    path.write_text(json.dumps(payload))


def test_no_regressions(tmp_path):
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    _write_suite(b, {"a": 0.8, "b": 0.7})
    _write_suite(c, {"a": 0.81, "b": 0.69})  # tiny moves only

    regs, imps = compare_suites(b, c, threshold=0.03)
    assert regs == []
    assert imps == []


def test_detects_regression(tmp_path):
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    _write_suite(b, {"a": 0.8, "b": 0.7})
    _write_suite(c, {"a": 0.6, "b": 0.7})  # 'a' dropped by 0.2

    regs, imps = compare_suites(b, c, threshold=0.03)
    assert len(regs) == 1
    assert regs[0].metric_name == "a"
    assert regs[0].delta < 0


def test_detects_improvement(tmp_path):
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    _write_suite(b, {"a": 0.5})
    _write_suite(c, {"a": 0.9})

    regs, imps = compare_suites(b, c, threshold=0.03)
    assert regs == []
    assert len(imps) == 1
    assert imps[0].delta > 0


def test_main_exits_nonzero_on_regression(tmp_path):
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    _write_suite(b, {"a": 0.9})
    _write_suite(c, {"a": 0.5})  # big drop

    exit_code = main(["--baseline", str(b), "--current", str(c), "--threshold", "0.03"])
    assert exit_code == 1


def test_main_exits_zero_on_clean(tmp_path):
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    _write_suite(b, {"a": 0.9})
    _write_suite(c, {"a": 0.91})

    exit_code = main(["--baseline", str(b), "--current", str(c), "--threshold", "0.03"])
    assert exit_code == 0
