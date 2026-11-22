import unittest

from grounding import get_topic_context
from prompt_builder import build_prompt
from validator import (
    extract_json,
    normalize_generated_io,
    run_test_cases,
    validate_structure,
)


class ProjectTests(unittest.TestCase):
    def test_grounding_uses_selected_difficulty(self):
        context = get_topic_context("Arrays", "Easy", "Python")
        self.assertIn("Suggested Constraints (Easy)", context)

    def test_prompt_uses_test_count_and_edge_case_selection(self):
        prompt = build_prompt("Arrays", "Easy", "Python", 4, True)
        self.assertIn("exactly 4 test cases", prompt[1]["content"])
        self.assertIn("at least one edge case", prompt[1]["content"])
        self.assertIn("standard input", prompt[1]["content"])
        self.assertEqual(prompt[1]["content"].count("<stdin for test case"), 4)

        prompt_without_edges = build_prompt("Arrays", "Easy", "Python", 2, False)
        self.assertIn("exactly 2 test cases", prompt_without_edges[1]["content"])
        self.assertIn("do not add a specifically chosen edge case", prompt_without_edges[1]["content"])

    def test_extracts_json_with_triple_quoted_solution(self):
        raw_output = (
            '```json\n{"title":"Example","reference_solution": """\n'
            'def solve():\n    print("ok")\n'
            '""", "test_cases": []}\n```'
        )

        parsed = extract_json(raw_output)

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["reference_solution"], '\ndef solve():\n    print("ok")\n')

    def test_validation_rejects_non_object_and_wrong_test_case_types(self):
        self.assertEqual(validate_structure([], "Python", 1), [
            "The model output must be a JSON object."
        ])

        malformed = {
            "title": "Example",
            "topic": "Arrays",
            "difficulty": "Easy",
            "problem_statement": "Statement",
            "constraints": [],
            "input_format": "Input",
            "output_format": "Output",
            "examples": [{"input": [1], "output": 1}],
            "reference_solution": "print(1)",
            "test_cases": [{"input": [1], "expected_output": 1}],
            "solution_explanation": "Explanation",
            "time_complexity": "O(1)",
            "space_complexity": "O(1)",
        }
        warnings = validate_structure(malformed, "Python", 1)
        self.assertTrue(any("Example 0" in warning for warning in warnings))
        self.assertTrue(any("Test case 0" in warning for warning in warnings))

    def test_normalizes_generated_array_and_numeric_io(self):
        problem = normalize_generated_io({
            "examples": [{"input": [1, -2], "output": 9}],
            "test_cases": [{"input": [], "expected_output": 0}],
        })

        self.assertEqual(problem["examples"][0], {
            "input": "1 -2",
            "output": "9",
        })
        self.assertEqual(problem["test_cases"][0], {
            "input": "",
            "expected_output": "0",
        })

    def test_test_runner_ignores_output_whitespace_only(self):
        problem = {
            "reference_solution": "print('1  2\\n3')",
            "test_cases": [{"input": "", "expected_output": "1 2 3\n"}],
        }
        result = run_test_cases(problem, "Python")
        self.assertTrue(result[0]["passed"])

        problem["test_cases"][0]["expected_output"] = "1 2 4"
        result = run_test_cases(problem, "Python")
        self.assertFalse(result[0]["passed"])

    def test_runs_python_solution_and_reports_malformed_cases(self):
        problem = {
            "reference_solution": "import sys\nprint(int(sys.stdin.read()) * 2)\n",
            "test_cases": [{"input": "4\n", "expected_output": "8"}],
        }
        result = run_test_cases(problem, "Python")
        self.assertTrue(result[0]["passed"])

        function_problem = {
            "reference_solution": (
                "def max_subarray_sum(nums):\n"
                "    best = current = nums[0]\n"
                "    for value in nums[1:]:\n"
                "        current = max(value, current + value)\n"
                "        best = max(best, current)\n"
                "    return best\n"
            ),
            "test_cases": [{"input": "1 -2 3 4", "expected_output": "7"}],
        }
        function_result = run_test_cases(function_problem, "Python")
        self.assertTrue(function_result[0]["passed"])

        list_result = run_test_cases({
            "reference_solution": "def solve(values):\n    return [value * 2 for value in values]\n",
            "test_cases": [{"input": "1 2 3", "expected_output": "2 4 6"}],
        }, "Python")
        self.assertTrue(list_result[0]["passed"])

        malformed = run_test_cases({
            "reference_solution": "print(1)",
            "test_cases": [None],
        }, "Python")
        self.assertFalse(malformed[0]["passed"])
        self.assertEqual(malformed[0]["error"], "Test case must be an object.")

        no_output = run_test_cases({
            "reference_solution": "result = 1",
            "test_cases": [{"input": "", "expected_output": "1"}],
        }, "Python")
        self.assertFalse(no_output[0]["passed"])
        self.assertIn("produced no output", no_output[0]["error"])


if __name__ == "__main__":
    unittest.main()
