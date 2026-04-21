"""Core data types for EvalForge.

An EvalCase is a single evaluation instance: an input, an expected answer,
retrieved context, and the model's actual output. An EvalResult carries the
scores that metrics produced for one case. An EvalSuite aggregates many cases.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalCase:
    """A single evaluation instance."""

    case_id: str
    question: str
    expected_answer: str
    actual_answer: str
    retrieved_context: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricScore:
    """A single metric's score on a single case."""

    metric_name: str
    score: float  # 0.0 to 1.0
    reasoning: str = ""
    samples: list[float] = field(default_factory=list)  # for self-consistency

    @property
    def std(self) -> float:
        """Standard deviation across self-consistency samples."""
        if len(self.samples) < 2:
            return 0.0
        return statistics.stdev(self.samples)


@dataclass
class EvalResult:
    """Result of evaluating a single case across multiple metrics."""

    case_id: str
    scores: dict[str, MetricScore] = field(default_factory=dict)

    def aggregate(self) -> float:
        """Mean score across all metrics."""
        if not self.scores:
            return 0.0
        return statistics.mean(s.score for s in self.scores.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "aggregate": self.aggregate(),
            "scores": {
                name: {
                    "score": s.score,
                    "reasoning": s.reasoning,
                    "std": s.std,
                    "samples": s.samples,
                }
                for name, s in self.scores.items()
            },
        }


@dataclass
class EvalSuite:
    """A collection of EvalResults with aggregation utilities."""

    name: str
    results: list[EvalResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, result: EvalResult) -> None:
        self.results.append(result)

    def metric_means(self) -> dict[str, float]:
        """Mean score per metric across all results."""
        if not self.results:
            return {}
        metric_names = set()
        for r in self.results:
            metric_names.update(r.scores.keys())

        means = {}
        for name in metric_names:
            values = [r.scores[name].score for r in self.results if name in r.scores]
            means[name] = statistics.mean(values) if values else 0.0
        return means

    def overall_score(self) -> float:
        """Mean of metric means — the headline number."""
        means = self.metric_means()
        if not means:
            return 0.0
        return statistics.mean(means.values())

    def pass_rate(self, threshold: float = 0.7) -> float:
        """Fraction of cases scoring >= threshold on aggregate."""
        if not self.results:
            return 0.0
        passing = sum(1 for r in self.results if r.aggregate() >= threshold)
        return passing / len(self.results)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "suite_name": self.name,
            "metadata": self.metadata,
            "overall_score": self.overall_score(),
            "pass_rate": self.pass_rate(),
            "metric_means": self.metric_means(),
            "results": [r.to_dict() for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def summary_table(self) -> str:
        """Human-readable summary."""
        lines = [f"=== EvalSuite: {self.name} ==="]
        lines.append(f"Total cases: {len(self.results)}")
        lines.append(f"Overall score: {self.overall_score():.3f}")
        lines.append(f"Pass rate (>=0.7): {self.pass_rate():.1%}")
        lines.append("")
        lines.append("Per-metric means:")
        for metric, mean in sorted(self.metric_means().items()):
            lines.append(f"  {metric:.<30} {mean:.3f}")
        return "\n".join(lines)
