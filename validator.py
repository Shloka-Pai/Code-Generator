"""
validator.py
------------
Validates LLM output and optionally executes the reference solution.

IMPORTANT DISCLAIMER:
  Executing LLM-generated code is inherently risky. The optional execution
  feature here uses subprocess with a timeout as a basic safeguard, but it
  is NOT a secure sandbox. Do NOT run this on untrusted inputs in production.
  For a college demo, it is acceptable with the understanding that you control
  the inputs. A proper sandbox would use Docker or a restricted VM.
"""

import ast
import json
import re
import subprocess
import sys
import tempfile
import os
from typing import Optional

# All fields the LLM must produce
REQUIRED_FIELDS = [
    "title",
    "topic",
    "difficulty",
    "problem_statement",
    "constraints",
    "input_format",
    "output_format",
    "examples",
    "reference_solution",
    "test_cases",
    "solution_explanation",
    "time_complexity",
    "space_complexity",
]

VALID_DIFFICULTIES = {"Easy", "Medium", "Hard"}
VALID_LANGUAGES = {"C++", "Python", "Java"}

TRIPLE_QUOTED_JSON_VALUE = re.compile(
    r"(?P<prefix>\"[^\"\n]+\"\s*:\s*)(?P<quote>\"\"\"|''')"
    r"(?P<value>.*?)(?P=quote)",
    re.DOTALL,
)


def normalize_generated_io(problem: dict) -> dict:
    """Convert generated JSON examples and test data to the text runner format."""
    if not isinstance(problem, dict):
        raise TypeError("Generated problem data must be a JSON object.")

    normalized = dict(problem)
    for field in ("examples", "test_cases"):
        entries = normalized.get(field)
        if not isinstance(entries, list):
            continue

        normalized_entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                normalized_entries.append(entry)
                continue

            normalized_entry = dict(entry)
            input_value = normalized_entry.get("input")
            if isinstance(input_value, list):
                if all(not isinstance(item, (list, dict)) for item in input_value):
                    normalized_entry["input"] = " ".join(map(str, input_value))
                else:
                    normalized_entry["input"] = "\n".join(
                        " ".join(map(str, row)) if isinstance(row, list) else str(row)
                        for row in input_value
                    )
            elif input_value is not None and not isinstance(input_value, str):
                normalized_entry["input"] = json.dumps(
                    input_value, ensure_ascii=False
                )

            output_field = "output" if field == "examples" else "expected_output"
            output_value = normalized_entry.get(output_field)
            if output_value is not None and not isinstance(output_value, str):
                if isinstance(output_value, list):
                    # Empty list [] must stay as "[]" not "" (join of empty list)
                    normalized_entry[output_field] = (
                        json.dumps(output_value) if len(output_value) == 0
                        else " ".join(map(str, output_value))
                    )
                else:
                    normalized_entry[output_field] = json.dumps(output_value, ensure_ascii=False)

            normalized_entries.append(normalized_entry)
        normalized[field] = normalized_entries

    return normalized


def _normalize_output(output: str) -> str:
    """Ignore insignificant whitespace differences as standard output judges do."""
    return " ".join(output.split())


def _add_function_solution_adapter(code: str) -> str:
    """Make common function-only Python solutions runnable with text test input."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    has_input = any(
        isinstance(node, ast.Name) and node.id == "input"
        or isinstance(node, ast.Attribute) and node.attr == "stdin"
        for node in ast.walk(tree)
    )
    has_print = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
        for node in ast.walk(tree)
    )
    if has_input and has_print:
        return code

    functions = [
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name != "main" and not node.name.startswith("_")
    ]
    if not functions:
        return code

    target = "solve" if "solve" in functions else functions[-1]
    return code + f"""

import inspect as __inspect
import json as __json
import sys as __sys

def __try_int(v):
    try: return int(v)
    except ValueError: return v

def __parse_token(t):
    try: return int(t)
    except ValueError:
        try: return float(t)
        except ValueError: return t

__raw_input = __sys.stdin.read().strip()
__lines = [l for l in __raw_input.splitlines() if l.strip()]
__tokens = __raw_input.split()
__function = {target}
__signature = __inspect.signature(__function)
__nparams = len(__signature.parameters)

# Build candidate argument sets
__argument_sets = []

# JSON parse (highest priority for JSON-formatted input)
try:
    __json_val = __json.loads(__raw_input)
    __argument_sets.append((__json_val,))
    if isinstance(__json_val, (list, tuple)):
        __argument_sets.append((list(__json_val),))
except (ValueError, TypeError):
    pass

# Single-param function: pass the whole token list as one argument
# If input has multiple lines and first line is a single int (likely n),
# try the remaining lines as the array first
if __nparams == 1:
    if len(__lines) >= 2:
        __rest_tokens = " ".join(__lines[1:]).split()
        try:
            __argument_sets.append(([int(t) for t in __rest_tokens],))
        except ValueError:
            __argument_sets.append((__rest_tokens,))
    try:
        __argument_sets.append(([int(t) for t in __tokens],))
    except ValueError:
        pass
    __argument_sets.append(([__parse_token(t) for t in __tokens],))
    __argument_sets.append((__raw_input,))
else:
    # Multi-param: each line becomes one argument
    if len(__lines) >= __nparams:
        __line_args = []
        for __line in __lines:
            __parts = __line.split()
            if len(__parts) == 1:
                __line_args.append(__try_int(__parts[0]))
            else:
                try: __line_args.append([int(x) for x in __parts])
                except ValueError: __line_args.append(__parts)
        __argument_sets.append(tuple(__line_args[:__nparams]))
        # Also try wrapping single-token lines as 1-element lists
        __line_args_wrapped = []
        for __a in __line_args[:__nparams]:
            __line_args_wrapped.append([__a] if isinstance(__a, int) else __a)
        if __line_args_wrapped != __line_args[:__nparams]:
            __argument_sets.append(tuple(__line_args_wrapped))
    # All tokens as individual int args
    try:
        __argument_sets.append(tuple(int(t) for t in __tokens))
    except ValueError:
        pass
    __argument_sets.append(tuple(__parse_token(t) for t in __tokens))
    __argument_sets.append((__raw_input,))

for __arguments in __argument_sets:
    try:
        __signature.bind(*__arguments)
    except TypeError:
        continue
    try:
        __result = __function(*__arguments)
    except TypeError:
        continue
    if __result is not None:
        if isinstance(__result, list) and all(
            not isinstance(__item, (list, dict)) for __item in __result
        ):
            print(" ".join(map(str, __result)))
        elif isinstance(__result, (list, dict)):
            print(__json.dumps(__result))
        else:
            print(__result)
    break
else:
    raise TypeError("Could not match test input to the solution function.")
"""


def _try_parse(text: str) -> Optional[dict]:
    """Try parsing JSON after removing trailing commas."""
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = TRIPLE_QUOTED_JSON_VALUE.sub(
        lambda m: m.group("prefix") + json.dumps(m.group("value"), ensure_ascii=False),
        text,
    )
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _recover_truncated(text: str) -> Optional[dict]:
    """Try to close a truncated JSON string by appending missing closers."""
    stack = []  # tracks '{' and '[' in order
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()

    if not stack and not in_string:
        return None  # Not truncated

    closing = ('"' if in_string else "")
    for opener in reversed(stack):
        closing += "}" if opener == "{" else "]"
    return _try_parse(text + closing)


def extract_json(raw_text: str) -> Optional[dict]:
    """
    Attempt to parse JSON from the LLM's raw output.
    Tries multiple strategies from strictest to most lenient.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text)
    cleaned = re.sub(r"```", "", cleaned).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    candidates = [raw_text.strip(), cleaned]
    if start != -1 and end > start:
        candidates.append(cleaned[start:end + 1])

    for candidate in candidates:
        result = _try_parse(candidate)
        if result:
            return result

    # Last resort: try to recover from truncated output
    fragment = cleaned[start:] if start != -1 else cleaned
    return _recover_truncated(fragment)


def validate_structure(problem: dict, language: str, num_test_cases: int) -> list[str]:
    """
    Check that the parsed problem dict has all required fields and basic consistency.

    Returns a list of warning strings (empty list = all checks passed).
    """
    if not isinstance(problem, dict):
        return ["The model output must be a JSON object."]

    warnings = []

    if language not in VALID_LANGUAGES:
        warnings.append(f"Unsupported language: '{language}'")

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in problem or problem[field] in (None, "", [], {}):
            warnings.append(f"Missing or empty field: '{field}'")

    string_fields = (
        "title", "topic", "problem_statement", "input_format", "output_format",
        "reference_solution", "solution_explanation", "time_complexity",
        "space_complexity",
    )
    for field in string_fields:
        if field in problem and not isinstance(problem[field], str):
            warnings.append(f"Field '{field}' must be a string.")

    if "constraints" in problem and not isinstance(problem["constraints"], list):
        warnings.append("Field 'constraints' must be a list.")

    # Check difficulty value
    if problem.get("difficulty") not in VALID_DIFFICULTIES:
        warnings.append(f"Unexpected difficulty: '{problem.get('difficulty')}'")

    # Check test case count — fewer than requested is a soft warning, not a hard error
    test_cases = problem.get("test_cases", [])
    if not isinstance(test_cases, list):
        warnings.append("Field 'test_cases' must be a list.")
    elif len(test_cases) == 0:
        warnings.append("Field 'test_cases' must not be empty.")

    # Check examples have required sub-fields
    examples = problem.get("examples", [])
    if not isinstance(examples, list):
        warnings.append("Field 'examples' must be a list.")
    else:
        for i, ex in enumerate(examples):
            if not isinstance(ex, dict):
                warnings.append(f"Example {i} is not a dict.")
                continue
            for key in ("input", "output"):
                if key not in ex or not isinstance(ex[key], str):
                    warnings.append(f"Example {i} must include '{key}' as a string.")

    # Check test cases have required sub-fields
    if isinstance(test_cases, list):
        for i, tc in enumerate(test_cases):
            if not isinstance(tc, dict):
                warnings.append(f"Test case {i} is not a dict.")
                continue
            for key in ("input", "expected_output"):
                if key not in tc or not isinstance(tc[key], str):
                    warnings.append(f"Test case {i} must include '{key}' as a string.")

    return warnings


def recompute_expected_outputs(problem: dict) -> tuple[dict, list[str]]:
    """
    Run the reference solution on every test case input and replace the
    LLM-generated expected_output with the actual output.
    Returns (updated_problem, list_of_warnings_for_cases_that_failed).
    """
    code = problem.get("reference_solution", "")
    test_cases = problem.get("test_cases", [])
    if not isinstance(code, str) or not code.strip() or not isinstance(test_cases, list):
        return problem, []

    updated = dict(problem)
    updated["test_cases"] = []
    warnings = []
    for i, tc in enumerate(test_cases):
        if not isinstance(tc, dict):
            updated["test_cases"].append(tc)
            continue
        stdin = tc.get("input", "")
        stdout, stderr = run_python_solution(code, stdin)
        new_tc = dict(tc)
        if stdout:
            new_tc["expected_output"] = stdout
        else:
            warnings.append(f"Test case {i+1}: reference solution produced no output ({stderr[:80] if stderr else 'no error'}).")
        updated["test_cases"].append(new_tc)
    return updated, warnings

def run_python_solution(code: str, input_data: str, timeout: int = 10) -> tuple[str, str]:
    """
    Execute a Python solution string with given stdin input.

    Uses subprocess so the generated code runs in a separate process.
    The timeout prevents infinite loops from hanging the app.

    WARNING: This is NOT a secure sandbox. Only use with code you trust.

    Returns (stdout_output, stderr_output).
    """
    # LLM often encodes newlines as literal \n inside the JSON string value.
    # After json.loads that becomes a real backslash + n, which is a syntax error
    # when written to a .py file.  Normalise before anything else.
    if "\\n" in code and "\n" not in code:
        code = code.replace("\\n", "\n")
    # Also normalise \t -> real tab
    code = code.replace("\\t", "\t")

    code = _add_function_solution_adapter(code)

    # Write code to a temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return "", f"Execution timed out after {timeout} seconds."
    except Exception as e:
        return "", str(e)
    finally:
        os.unlink(tmp_path)  # Clean up temp file


def run_test_cases(problem: dict, language: str) -> list[dict]:
    """
    Run the reference solution against all test cases (Python only).

    Returns a list of result dicts with pass/fail status.

    DISCLAIMER: Even if tests pass, correctness is not guaranteed because
    the expected outputs were also generated by the LLM and may be wrong.
    Independent verification is required for academic or production use.
    """
    if language != "Python":
        return [{"note": f"Automated execution is only supported for Python. Got: {language}"}]

    if not isinstance(problem, dict):
        return [{"note": "Problem data must be a JSON object."}]

    code = problem.get("reference_solution", "")
    test_cases = problem.get("test_cases", [])

    if not isinstance(code, str) or not code.strip():
        return [{"note": "No reference solution found."}]
    if not isinstance(test_cases, list):
        return [{"note": "Test cases must be a list."}]

    results = []
    for i, tc in enumerate(test_cases):
        if not isinstance(tc, dict):
            results.append({
                "test_case": i + 1,
                "passed": False,
                "error": "Test case must be an object.",
            })
            continue

        stdin_input = tc.get("input", "")
        expected_value = tc.get("expected_output", "")
        if not isinstance(stdin_input, str) or not isinstance(expected_value, str):
            results.append({
                "test_case": i + 1,
                "passed": False,
                "error": "Test case input and expected_output must be strings.",
            })
            continue
        expected = expected_value.strip()

        # LLM sometimes generates empty input for edge cases (e.g. empty array)
        # but the solution still calls input() expecting at least "0".
        # Try running with "0" as a best-effort fallback before giving up.
        if not stdin_input.strip() and expected:
            stdout, stderr = run_python_solution(code, "0")
            if stdout:
                stdin_input = "0"
            else:
                results.append({
                    "test_case": i + 1,
                    "input": stdin_input,
                    "expected": expected,
                    "got": "",
                    "passed": False,
                    "error": "Test case has empty input — the LLM generated an incomplete edge case. Verify manually.",
                })
                continue

        stdout, stderr = run_python_solution(code, stdin_input)
        if not stdout and not stderr and expected:
            stderr = (
                "Solution produced no output. Make sure it prints the expected "
                "answer or returns it from a supported function."
            )

        passed = _normalize_output(stdout) == _normalize_output(expected)
        results.append({
            "test_case": i + 1,
            "input": stdin_input,
            "expected": expected,
            "got": stdout,
            "passed": passed,
            "error": stderr if stderr else None,
        })

    return results
