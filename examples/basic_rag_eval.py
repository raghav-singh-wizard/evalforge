"""Basic RAG evaluation example.

Shows how to:
  1. Define EvalCases manually (this is what you'd do for your own RAG outputs)
  2. Pick metrics
  3. Run the evaluation
  4. Save a JSON suite and print a summary

Run with:   make example
        or: python examples/basic_rag_eval.py
"""

from __future__ import annotations

from evalforge import (
    AnswerRelevance,
    Faithfulness,
    Groundedness,
    HallucinationRate,
    run_evaluation,
)
from evalforge.core import EvalCase
from evalforge.judges.mock import MockJudge


def main() -> None:
    # --- Step 1: define your cases ---
    cases = [
        EvalCase(
            case_id="demo_001",
            question="What is the boiling point of water at sea level?",
            expected_answer="100 degrees Celsius (212 Fahrenheit)",
            actual_answer="Water boils at 100°C at sea level.",
            retrieved_context=[
                "The boiling point of water at standard atmospheric pressure (sea level) is 100 °C.",
            ],
        ),
        EvalCase(
            case_id="demo_002",
            question="Who painted the Mona Lisa?",
            expected_answer="Leonardo da Vinci",
            actual_answer="The Mona Lisa was painted by Leonardo da Vinci in the early 1500s.",
            retrieved_context=[
                "The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci.",
            ],
        ),
        EvalCase(
            case_id="demo_003",
            # A deliberate hallucination — answer claims something not in context
            question="How many moons does Mars have?",
            expected_answer="Two — Phobos and Deimos",
            actual_answer="Mars has three moons: Phobos, Deimos, and Ares.",
            retrieved_context=[
                "Mars has two natural satellites: Phobos and Deimos.",
            ],
        ),
    ]

    # --- Step 2: pick a judge and metrics ---
    judge = MockJudge(n_samples=2)  # swap for OpenAIJudge / AnthropicJudge in real runs
    metrics = [
        AnswerRelevance(judge),
        Groundedness(judge),
        Faithfulness(judge),
        HallucinationRate(judge),
    ]

    # --- Step 3: run ---
    suite = run_evaluation(cases, metrics, suite_name="basic_demo", max_workers=2)

    # --- Step 4: report ---
    print(suite.summary_table())
    suite.save_json("benchmarks/results/example_output.json")
    print("\nSaved to benchmarks/results/example_output.json")


if __name__ == "__main__":
    main()
