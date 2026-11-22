"""
grounding.py
------------
Retrieves relevant DSA context from problem_bank.json for a given topic.

WHY GROUNDING?
  The LLM may hallucinate incorrect complexities or constraints.
  By injecting verified facts from our local knowledge base into the prompt,
  we guide the model toward accurate outputs — this is the core idea of RAG
  (Retrieval-Augmented Generation), done here without any vector database.

LIMITATION: Grounding reduces hallucinations but does NOT guarantee correctness.
  The LLM can still ignore or misuse the provided context.
"""

import json
import os

# Path to the knowledge base file
KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(__file__), "problem_bank.json")

def load_knowledge_base() -> dict:
    """Load and return the full knowledge base from JSON."""
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def get_topic_context(topic: str, difficulty: str, language: str) -> str:
    """
    Retrieve a concise grounding string for the given topic, difficulty, and language.
    This string is injected into the LLM prompt to anchor its output in verified facts.

    Returns a plain-text summary the LLM can read and use.
    """
    kb = load_knowledge_base()
    topics = kb.get("topics", {})
    languages = kb.get("languages", {})
    difficulty_guidelines = kb.get("difficulty_guidelines", {})

    # --- Topic context ---
    topic_data = topics.get(topic, {})
    if not topic_data:
        # Fallback: return minimal context if topic not found
        return f"Topic: {topic}\nDifficulty: {difficulty}\nLanguage: {language}"

    lines = []
    lines.append(f"=== GROUNDING CONTEXT FOR PROMPT ===")
    lines.append(f"Topic: {topic}")
    lines.append(f"Description: {topic_data.get('description', '')}")

    concepts = topic_data.get("concepts", [])
    if concepts:
        lines.append(f"Key Concepts: {', '.join(concepts)}")

    # Constraint guidelines for the selected difficulty
    constraints = topic_data.get("constraint_guidelines", {})
    diff_key = difficulty.lower()
    if diff_key in constraints:
        lines.append(f"Suggested Constraints ({difficulty}): {constraints[diff_key]}")

    # Typical complexities
    complexities = topic_data.get("typical_complexities", {})
    if diff_key in complexities:
        lines.append(f"Expected Complexity ({difficulty}): {complexities[diff_key]}")

    # Edge cases
    edge_cases = topic_data.get("edge_cases", [])
    if edge_cases:
        lines.append(f"Common Edge Cases: {'; '.join(edge_cases)}")

    # Difficulty guideline
    diff_guide = difficulty_guidelines.get(difficulty, "")
    if diff_guide:
        lines.append(f"Difficulty Guideline: {diff_guide}")

    # Language notes
    lang_data = languages.get(language, {})
    if lang_data:
        lines.append(f"Language Notes ({language}): {lang_data.get('notes', '')}")

    lines.append("=== END GROUNDING CONTEXT ===")
    return "\n".join(lines)


def get_available_topics() -> list:
    """Return list of all available topics from the knowledge base."""
    kb = load_knowledge_base()
    return list(kb.get("topics", {}).keys())


def get_available_languages() -> list:
    """Return list of supported programming languages."""
    kb = load_knowledge_base()
    return list(kb.get("languages", {}).keys())
