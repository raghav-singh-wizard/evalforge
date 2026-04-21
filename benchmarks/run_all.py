"""Run EvalForge across the five built-in benchmark samples.

Used by `make eval`. Produces a single JSON suite file combining results
from HotpotQA, TriviaQA, MMLU-Pro, TruthfulQA, and FinanceBench.

The default judge is MockJudge (no API key needed, CI-friendly). Swap in
OpenAIJudge or AnthropicJudge for real runs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running from repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalforge import (
    AnswerRelevance,
    Faithfulness,
    Groundedness,
    HallucinationRate,
    run_evaluation,
)
from evalforge.core import EvalSuite
from evalforge.datasets import (
    load_financebench_sample,
    load_hotpotqa_sample,
    load_mmlu_pro_sample,
    load_triviaqa_sample,
    load_truthfulqa_sample,
)
from evalforge.judges.mock import MockJudge


def build_judge(judge_name: str, samples: int, model: str | None):
    if judge_name == "mock":
        return MockJudge(n_samples=samples, temperature=0.7)
    if judge_name == "openai":
        from evalforge.judges.openai_judge import OpenAIJudge

        return OpenAIJudge(model=model or "gpt-4o-mini", n_samples=samples)
    if judge_name == "anthropic":
        from evalforge.judges.anthropic_judge import AnthropicJudge

        return AnthropicJudge(
            model=model or "claude-3-5-sonnet-latest", n_samples=samples
        )
    raise ValueError(f"Unknown judge: {judge_name}")


DATASETS = {
    "hotpotqa": load_hotpotqa_sample,
    "triviaqa": load_triviaqa_sample,
    "mmlu_pro": load_mmlu_pro_sample,
    "truthfulqa": load_truthfulqa_sample,
    "financebench": load_financebench_sample,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EvalForge full benchmark.")
    parser.add_argument("--output", default="benchmarks/results/current.json")
    parser.add_argument(
        "--judge",
        default=os.environ.get("EVALFORGE_JUDGE", "mock"),
        choices=["mock", "openai", "anthropic"],
    )
    parser.add_argument("--model", default=None, help="Override judge model name")
    parser.add_argument("--samples", type=int, default=2, help="Self-consistency samples")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
    )
    args = parser.parse_args()

    judge = build_judge(args.judge, args.samples, args.model)
    metrics = [
        AnswerRelevance(judge),
        Groundedness(judge),
        Faithfulness(judge),
        HallucinationRate(judge),
    ]

    print(f"=== EvalForge benchmark ===")
    print(f"Judge:    {args.judge} ({judge.model})")
    print(f"Samples:  {args.samples}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print()

    combined = EvalSuite(
        name="evalforge_full",
        metadata={
            "judge": args.judge,
            "model": judge.model,
            "samples": args.samples,
            "datasets": args.datasets,
        },
    )

    for ds_name in args.datasets:
        print(f"--- {ds_name} ---")
        cases = DATASETS[ds_name]()
        suite = run_evaluation(
            cases,
            metrics,
            suite_name=ds_name,
            max_workers=args.workers,
            verbose=True,
        )
        for r in suite.results:
            # Tag case ids so they don't collide across datasets in the combined suite
            r.case_id = f"{ds_name}/{r.case_id}"
            combined.add(r)
        print(f"  -> overall: {suite.overall_score():.3f}  pass@0.7: {suite.pass_rate():.1%}")
        print()

    out_path = Path(args.output)
    combined.save_json(out_path)
    print("=" * 40)
    print(combined.summary_table())
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
