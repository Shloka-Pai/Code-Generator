"""
evaluation.py
-------------
Measures LLM output quality across multiple runs and temperature settings.

Run from the terminal:
    python evaluation.py

This script does NOT invent results — it actually calls the model and
measures real metrics. Results are saved to evaluation_results.json.

Metrics measured:
  - valid_json_rate       : % of outputs that are parseable JSON
  - all_fields_rate       : % of outputs with all required fields present
  - avg_generation_time   : average seconds per generation
  - avg_token_count       : average number of generated tokens
  - test_pass_rate        : % of Python test cases that pass (LLM-generated expected outputs)
                            NOTE: this measures internal consistency, not true correctness.

Experiments:
  Compares temperatures [0.1, 0.3, 0.7] to show how temperature affects output quality.
"""

import json
import time
from model import generate_problem, GENERATION_CONFIG
from prompt_builder import build_prompt
from validator import (
    extract_json,
    normalize_generated_io,
    run_test_cases,
    validate_structure,
)

# ── Experiment configuration ──────────────────────────────────────────────────
TEMPERATURES_TO_TEST = [0.1, 0.3, 0.7]
RUNS_PER_TEMPERATURE = 3   # Increase for more reliable averages (takes longer)

# Fixed inputs for fair comparison across temperatures
EVAL_TOPIC = "Arrays"
EVAL_DIFFICULTY = "Easy"
EVAL_LANGUAGE = "Python"
EVAL_NUM_TEST_CASES = 2
EVAL_INCLUDE_EDGE_CASES = False

OUTPUT_FILE = "evaluation_results.json"


def run_single_experiment(temperature: float) -> dict:
    """Run one generation with the given temperature and return metrics."""
    messages = build_prompt(
        topic=EVAL_TOPIC,
        difficulty=EVAL_DIFFICULTY,
        language=EVAL_LANGUAGE,
        num_test_cases=EVAL_NUM_TEST_CASES,
        include_edge_cases=EVAL_INCLUDE_EDGE_CASES,
    )

    raw_text, elapsed, token_count = generate_problem(
        messages,
        config_overrides={"temperature": temperature},
    )

    # Try to parse JSON
    problem = extract_json(raw_text)
    valid_json = problem is not None
    if problem is not None:
        problem = normalize_generated_io(problem)

    # Check field completeness
    all_fields_ok = False
    warnings = []
    if valid_json:
        warnings = validate_structure(problem, EVAL_LANGUAGE, EVAL_NUM_TEST_CASES)
        all_fields_ok = len(warnings) == 0

    # Run test cases if JSON is valid and language is Python
    test_results = []
    if valid_json:
        test_results = run_test_cases(problem, EVAL_LANGUAGE)

    # Count passing test cases (only meaningful if expected outputs are correct)
    passing = sum(1 for r in test_results if r.get("passed", False))
    total_tests = sum(1 for r in test_results if "passed" in r)

    return {
        "temperature": temperature,
        "valid_json": valid_json,
        "all_fields_present": all_fields_ok,
        "warnings": warnings,
        "generation_time_seconds": round(elapsed, 2),
        "token_count": token_count,
        "test_cases_passed": passing,
        "test_cases_total": total_tests,
        "raw_output_snippet": raw_text[:300],  # First 300 chars for inspection
    }


def run_evaluation():
    """Run all experiments and save results to JSON."""
    print("=" * 60)
    print("DSA LLM Generator — Evaluation Script")
    print("=" * 60)
    print(f"Topic: {EVAL_TOPIC} | Difficulty: {EVAL_DIFFICULTY} | Language: {EVAL_LANGUAGE}")
    print(f"Temperatures: {TEMPERATURES_TO_TEST} | Runs per temp: {RUNS_PER_TEMPERATURE}")
    print("=" * 60)

    all_results = []
    summary = {}

    for temp in TEMPERATURES_TO_TEST:
        print(f"\n[Temperature = {temp}]")
        temp_results = []

        for run_idx in range(RUNS_PER_TEMPERATURE):
            print(f"  Run {run_idx + 1}/{RUNS_PER_TEMPERATURE} ...", end=" ", flush=True)
            result = run_single_experiment(temp)
            temp_results.append(result)
            all_results.append(result)
            print(
                f"valid_json={result['valid_json']} | "
                f"all_fields={result['all_fields_present']} | "
                f"time={result['generation_time_seconds']}s | "
                f"tokens={result['token_count']}"
            )

        # Aggregate metrics for this temperature
        n = len(temp_results)
        summary[str(temp)] = {
            "valid_json_rate": sum(r["valid_json"] for r in temp_results) / n,
            "all_fields_rate": sum(r["all_fields_present"] for r in temp_results) / n,
            "avg_generation_time": round(
                sum(r["generation_time_seconds"] for r in temp_results) / n, 2
            ),
            "avg_token_count": round(
                sum(r["token_count"] for r in temp_results) / n, 1
            ),
        }

    # Save everything to file
    output = {
        "config": {
            "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "topic": EVAL_TOPIC,
            "difficulty": EVAL_DIFFICULTY,
            "language": EVAL_LANGUAGE,
            "runs_per_temperature": RUNS_PER_TEMPERATURE,
        },
        "summary_by_temperature": summary,
        "all_runs": all_results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for temp, metrics in summary.items():
        print(f"\nTemperature {temp}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    print(f"\nFull results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_evaluation()
