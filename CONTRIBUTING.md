# Contributing to EvalForge

Thanks for your interest. This is a small project — contributions welcome.

## Setup

```bash
git clone https://github.com/raghav-singh-wizard/evalforge
cd evalforge
make install-dev
```

## Before you open a PR

```bash
make lint    # ruff + mypy
make test    # pytest
make eval    # run the benchmark
```

All three must pass. The CI runs the same steps plus the regression gate on PRs.

## Adding a new metric

1. Subclass `evalforge.metrics.base.BaseMetric`.
2. Implement `_build_prompt(case: EvalCase) -> str`.
3. Add a test in `tests/test_metrics.py`.

See `evalforge/metrics/groundedness.py` for a concrete example.

## Adding a new judge backend

1. Subclass `evalforge.judges.base.BaseJudge`.
2. Implement `_call(prompt: str, temperature: float) -> str` — the rest (self-consistency sampling, parsing) is handled in the base class.
3. Make imports lazy inside `_get_client()` so users who don't install the backend's SDK aren't forced to.

See `evalforge/judges/openai_judge.py`.

## Style

- Ruff + mypy enforced.
- Line length 100.
- Type hints required on all public APIs.
- Tests for every new public function.

## Questions

Open an issue. I read them.
