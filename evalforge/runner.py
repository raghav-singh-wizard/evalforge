"""Evaluation runner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from evalforge.core import EvalCase, EvalResult, EvalSuite
from evalforge.metrics.base import BaseMetric


def run_evaluation(
    cases: Iterable[EvalCase],
    metrics: list[BaseMetric],
    suite_name: str = "default",
    max_workers: int = 4,
    verbose: bool = False,
) -> EvalSuite:
    """Run each metric on each case, return aggregated EvalSuite.

    Parallelism is per-case; metrics on the same case run sequentially to keep
    judge-call ordering readable in logs.
    """
    cases_list = list(cases)
    suite = EvalSuite(name=suite_name)

    def _score_one(case: EvalCase) -> EvalResult:
        result = EvalResult(case_id=case.case_id)
        for metric in metrics:
            s = metric.score(case)
            result.scores[metric.name] = s
        if verbose:
            print(f"  [{case.case_id}] aggregate={result.aggregate():.3f}")
        return result

    if max_workers <= 1:
        for case in cases_list:
            suite.add(_score_one(case))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_score_one, c): c.case_id for c in cases_list}
            results_by_id = {}
            for fut in as_completed(futures):
                case_id = futures[fut]
                results_by_id[case_id] = fut.result()
            # Preserve input order
            for case in cases_list:
                suite.add(results_by_id[case.case_id])

    return suite
