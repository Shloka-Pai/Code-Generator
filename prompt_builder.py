"""
prompt_builder.py
-----------------
Builds the structured prompt sent to the LLM.

The prompt uses a system message + user message format compatible with
Qwen2.5-Coder-Instruct's chat template.

The LLM is instructed to return ONLY valid JSON — no extra text.
This makes parsing reliable and avoids post-processing headaches.
"""

from grounding import get_topic_context

SYSTEM_PROMPT = (
    "You are a DSA problem generator. "
    "Respond with ONLY one valid JSON object. Do not use markdown or code fences. "
    "Every value must follow the JSON schema and types requested. "
    "The reference_solution must always use input() and print(), never be a bare function."
)


def build_prompt(
    topic: str,
    difficulty: str,
    language: str,
    num_test_cases: int,
    include_edge_cases: bool,
) -> list[dict]:
    # Get key facts from knowledge base (constraints, complexity, edge cases)
    grounding_context = get_topic_context(topic, difficulty, language)

    edge_case_instruction = (
        "Include at least one edge case among the test cases."
        if include_edge_cases
        else "Use ordinary cases; do not add a specifically chosen edge case."
    )
    # Build explicit numbered test case slots so the LLM cannot stop early
    test_case_slots = ",\n".join(
        f'    {{"input": "<test {index} stdin — must not be empty>", '
        f'"expected_output": "<test {index} stdout — must not be empty>"}}'
        for index in range(1, num_test_cases + 1)
    )

    user_message = f"""Generate a {difficulty} {topic} DSA problem with a {language} solution.

Facts:
{grounding_context}

Rules:
- Return ONLY a single valid JSON object, no text outside it.
- All string values use double quotes. Escape newlines in code as \\n, tabs as \\t.
- The reference_solution must use input() to read stdin and print() to write stdout. Example: n=int(input()); arr=list(map(int,input().split())); print(max(arr)). Never write a bare function with no input/print.
- problem_statement: 2-3 sentences max.
- solution_explanation: 1-2 sentences max.
- constraints: at most 3 items.
- examples: exactly 1 item.
- test_cases: you MUST fill all {num_test_cases} slots below. Each input and expected_output must be a non-empty string. {edge_case_instruction} For edge cases like empty arrays use n=0 on the first line.

JSON shape — fill every field, replace every placeholder, do not omit any test_cases slot:
{{
  "title": "short title",
  "topic": "{topic}",
  "difficulty": "{difficulty}",
  "problem_statement": "...",
  "input_format": "...",
  "output_format": "...",
  "constraints": ["..."],
  "examples": [{{"input": "...", "output": "...", "explanation": "..."}}],
  "reference_solution": "...",
  "test_cases": [
{test_case_slots}
  ],
  "solution_explanation": "...",
  "time_complexity": "O(?)",
  "space_complexity": "O(?)"
}}"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
