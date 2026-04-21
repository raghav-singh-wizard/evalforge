"""Dataset adapters for public QA benchmarks.

Each loader returns a list of EvalCase. The ones here cover the datasets
claimed on the resume: HotpotQA, TriviaQA, MMLU-Pro, TruthfulQA, FinanceBench.

For CI speed, we use tiny built-in sample slices. Real runs can swap in the
full HuggingFace datasets via `load_from_hf()`.
"""

from evalforge.datasets.loaders import (
    load_financebench_sample,
    load_from_hf,
    load_hotpotqa_sample,
    load_mmlu_pro_sample,
    load_truthfulqa_sample,
    load_triviaqa_sample,
)

__all__ = [
    "load_hotpotqa_sample",
    "load_triviaqa_sample",
    "load_mmlu_pro_sample",
    "load_truthfulqa_sample",
    "load_financebench_sample",
    "load_from_hf",
]
