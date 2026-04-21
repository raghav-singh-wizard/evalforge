# EvalForge

**Open-source LLM evaluation harness with pytest integration and regression-testing CI.**

EvalForge measures groundedness, faithfulness, answer relevance, and hallucination rate for LLM-powered pipelines — and catches silent quality regressions on every pull request before they ship to production.

[![CI](https://github.com/raghav-singh-wizard/evalforge/actions/workflows/evalforge-ci.yml/badge.svg)](https://github.com/raghav-singh-wizard/evalforge/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## The problem

Every LLM team has shipped a prompt change or model upgrade that silently broke quality and only noticed when users complained. Traditional software testing doesn't catch this — LLM outputs are non-deterministic, and "did the response look good" is not an assertion.

EvalForge turns "does my LLM still work" into a pytest assertion and a PR status check.

## What it does

- **Four built-in metrics.** Groundedness (is the answer supported by context?), Faithfulness (does it match the gold answer?), Answer Relevance (is it on-topic?), Hallucination Rate (did the model invent claims?).
- **LLM-as-a-judge with self-consistency sampling.** Every metric runs the judge `N` times and aggregates, stabilizing scores against LLM non-determinism.
- **Multiple judge backends.** OpenAI, Anthropic, or a deterministic `MockJudge` for offline CI.
- **pytest plugin.** Write eval suites as regular pytest tests. Fail the suite if `suite.overall_score() < 0.7`.
- **Regression gate.** Compares your PR branch's scores against a baseline from `main`. Blocks the PR if any metric dropped by more than 3%.
- **Benchmark reproducibility.** `make eval` runs all five bundled benchmark samples (HotpotQA, TriviaQA, MMLU-Pro, TruthfulQA, FinanceBench) with a single command.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        EvalForge                             │
│                                                              │
│   ┌───────────┐      ┌───────────┐      ┌──────────────┐    │
│   │ EvalCase  │─────▶│  Metric   │─────▶│ MetricScore  │    │
│   └───────────┘      └─────┬─────┘      └──────────────┘    │
│                            │                                 │
│                            ▼                                 │
│                      ┌───────────┐                          │
│                      │   Judge   │  (OpenAI / Anthropic /   │
│                      │  (N-samp) │   Mock)                   │
│                      └───────────┘                          │
│                                                              │
│   ┌──────────────────────────────────────────────────────┐  │
│   │                   EvalSuite                          │  │
│   │  overall_score · metric_means · pass_rate · JSON     │  │
│   └──────────────────────────────────────────────────────┘  │
│                            │                                 │
│                            ▼                                 │
│   ┌──────────────────────────────────────────────────────┐  │
│   │    Regression Gate   (baseline.json vs current.json) │  │
│   │    ↓ fails CI if any metric dropped > 3%             │  │
│   └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Quickstart

```bash
git clone https://github.com/raghav-singh-wizard/evalforge
cd evalforge
pip install -e ".[dev]"

# Run tests
make test

# Run the full benchmark suite (uses MockJudge — no API key needed)
make eval

# Promote current results as the baseline for future runs
make baseline

# Simulate a PR: re-run eval and check for regressions
make eval
make gate
```

Expected output from `make eval`:

```
=== EvalForge benchmark ===
Judge:    mock (mock)
Samples:  2
Datasets: hotpotqa, triviaqa, mmlu_pro, truthfulqa, financebench

--- hotpotqa ---
  [hotpot_001] aggregate=0.673
  [hotpot_002] aggregate=0.701
  -> overall: 0.687  pass@0.7: 50.0%
...
========================================
=== EvalSuite: evalforge_full ===
Total cases: 10
Overall score: 0.692
Pass rate (>=0.7): 60.0%

Per-metric means:
  answer_relevance.............. 0.751
  faithfulness.................. 0.683
  groundedness.................. 0.702
  hallucination_rate............ 0.632
```

## Use it in your own codebase

```python
from evalforge import AnswerRelevance, Groundedness, run_evaluation
from evalforge.core import EvalCase
from evalforge.judges.anthropic_judge import AnthropicJudge

judge = AnthropicJudge(model="claude-3-5-sonnet-latest", n_samples=3)

cases = [
    EvalCase(
        case_id="user_q_001",
        question="What is our Q3 return policy?",
        expected_answer="30-day full refund, excluding consumables.",
        actual_answer=my_rag_pipeline("What is our Q3 return policy?"),
        retrieved_context=my_retriever.get_context(...),
    ),
    # ... more cases
]

suite = run_evaluation(
    cases,
    metrics=[AnswerRelevance(judge), Groundedness(judge)],
    suite_name="return_policy_rag",
)

assert suite.overall_score() >= 0.7, suite.summary_table()
```

## pytest integration

EvalForge ships a pytest plugin. Drop it into your test suite:

```python
# tests/test_rag.py
from evalforge import AnswerRelevance, Groundedness, run_evaluation
from evalforge.datasets import load_hotpotqa_sample


def test_hotpot_quality(mock_judge):  # fixture auto-provided
    metrics = [AnswerRelevance(mock_judge), Groundedness(mock_judge)]
    suite = run_evaluation(load_hotpotqa_sample(), metrics, suite_name="hotpot")

    assert suite.overall_score() >= 0.6
    assert suite.pass_rate(threshold=0.5) >= 0.75
```

Run with `pytest` — failing assertions are just test failures, so any CI system (not just GitHub Actions) works.

## The regression gate

Every PR runs the full eval suite and compares scores to `main`'s baseline:

```
=== Regression Gate Report ===

❌ 1 regression(s):
  ↓ groundedness             baseline=0.820 current=0.740 delta=-0.080
```

The gate exits non-zero, the PR status turns red, and a comment is posted on the PR. The threshold is configurable (default: 3% absolute drop).

## Judge backends

| Backend        | When to use                          | Setup                                 |
| -------------- | ------------------------------------ | ------------------------------------- |
| `MockJudge`    | CI, tests, offline dev               | None — built in                       |
| `OpenAIJudge`  | Real evaluation with GPT-4 family    | `pip install openai`, `OPENAI_API_KEY`|
| `AnthropicJudge` | Real evaluation with Claude family | `pip install anthropic`, `ANTHROPIC_API_KEY` |

Add a custom judge by subclassing `BaseJudge` and implementing `_call(prompt, temperature)`.

## Why self-consistency sampling?

LLMs are non-deterministic at `temperature > 0`. A single judge call on the same input can return 0.6, 0.8, or 0.5 on different runs. EvalForge runs the judge `N` times (default: 3) — the first call at `temperature=0` for a stable anchor, subsequent calls at the configured temperature to sample variance. The final score is the mean, and the per-sample standard deviation is kept on the `MetricScore` so you can tell "the judge is confident" from "the judge is flip-flopping."

## Project layout

```
evalforge/
├── evalforge/
│   ├── core.py              # EvalCase, EvalResult, EvalSuite, MetricScore
│   ├── runner.py            # run_evaluation()
│   ├── regression_gate.py   # CI gate (also a CLI: evalforge-gate)
│   ├── pytest_plugin.py     # pytest fixtures
│   ├── metrics/             # AnswerRelevance, Groundedness, Faithfulness, HallucinationRate
│   ├── judges/              # BaseJudge, MockJudge, OpenAIJudge, AnthropicJudge
│   └── datasets/            # Loaders for 5 public benchmarks + HF fallback
├── benchmarks/
│   └── run_all.py           # Script invoked by `make eval`
├── examples/
│   └── basic_rag_eval.py    # Minimal end-to-end example
├── tests/                   # pytest test suite (55+ tests, ~85% coverage)
└── .github/workflows/
    └── evalforge-ci.yml     # Test matrix + regression gate
```

## Limitations — what EvalForge does NOT do

Being honest about scope:

- **LLM-as-a-judge inherits judge bias.** If the judge model is worse than the model under evaluation, scores are noisy. Mitigate with self-consistency, multiple judges, or human spot-checks. The `MockJudge` is a harness test, not a real evaluator.
- **Metrics are surface-level.** Groundedness asks "is each claim in context?" — it can miss compositional errors where each claim is grounded but the combination is wrong.
- **No latency/cost tracking yet.** The headline metrics are quality-only. Production deployments need latency, token cost, and retrieval-hit-rate instrumentation too. `v0.2` planned.
- **Bundled benchmark samples are tiny** (2 cases each). For real runs use `load_from_hf()` to pull the full datasets from HuggingFace.

## License

MIT — see [LICENSE](LICENSE).

## Author

[Raghubir Kumar Singh](https://github.com/raghav-singh-wizard) · [LinkedIn](https://www.linkedin.com/in/raghubir-kr-singh/)

Built because I kept shipping LLM changes at work and watching quality silently drift between deploys. Every metric, prompt, and design choice here is something I wish existed at my previous jobs.
