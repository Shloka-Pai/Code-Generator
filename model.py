"""
model.py
--------
Handles loading Qwen2.5-Coder-1.5B-Instruct and running inference.

The model is downloaded from Hugging Face on first run and cached locally
at ~/.cache/huggingface/hub/ — subsequent runs load from cache (offline-capable).

MEMORY REQUIREMENTS:
  - FP32: ~6 GB RAM  (not recommended)
  - FP16: ~3 GB RAM  (recommended, GPU or CPU)
  - 4-bit quantized: ~1 GB RAM (requires bitsandbytes)
  The default here uses FP16 on GPU if available, else FP32 on CPU.
"""

import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set to "1" to block all network calls after the model is downloaded
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ── Model identifier on Hugging Face Hub ──────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# ── Generation hyperparameters — change these to experiment ───────────────────
GENERATION_CONFIG = {
    "max_new_tokens": 1500,   # Full JSON with solution + test cases needs ~1000-1400 tokens
    "temperature": 0.3,       # Lower = more deterministic, higher = more creative
    "top_p": 0.9,             # Nucleus sampling: keep tokens covering 90% probability mass
    "do_sample": True,        # Must be True to use temperature/top_p
    "repetition_penalty": 1.1,  # Slightly penalise repeated phrases
}

# Module-level cache so the model loads only once per session
_tokenizer = None
_model = None


def load_model():
    """
    Load tokenizer and model into memory.
    Called once; subsequent calls return the cached objects.

    Uses float16 on CUDA GPU for speed and memory efficiency.
    Falls back to float32 on CPU (slower but works on any machine).
    """
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model  # Already loaded

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[model.py] Loading {MODEL_ID} on {device} with dtype={dtype} ...")
    print("[model.py] First run will download ~3 GB. Subsequent runs use local cache.")

    try:
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,  # Required for Qwen models
        )

        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=dtype,
            device_map={"": 0} if device == "cuda" else {"": "cpu"},
            trust_remote_code=True,
        )
        _model.eval()  # Disable dropout — we are doing inference, not training
        print(f"[model.py] Model loaded successfully.")

    except Exception as e:
        raise RuntimeError(
            f"Failed to load model '{MODEL_ID}'.\n"
            f"Error: {e}\n\n"
            "Possible fixes:\n"
            "  1. Check your internet connection for first-time download.\n"
            "  2. Ensure you have ~3 GB free RAM (FP16) or ~6 GB (FP32).\n"
            "  3. Install dependencies: pip install transformers torch accelerate\n"
            "  4. For low-memory machines, try 4-bit: pip install bitsandbytes\n"
            "     then set load_in_4bit=True in from_pretrained()."
        )

    return _tokenizer, _model


def generate_problem(messages: list[dict], config_overrides: dict = None) -> tuple[str, float, int]:
    """
    Run inference given a list of chat messages.

    Args:
        messages: List of {"role": ..., "content": ...} dicts from prompt_builder.
        config_overrides: Optional dict to override GENERATION_CONFIG values.

    Returns:
        (generated_text, elapsed_seconds, token_count)
    """
    tokenizer, model = load_model()

    # Merge default config with any overrides (used by evaluation.py)
    gen_config = {**GENERATION_CONFIG, **(config_overrides or {})}

    # apply_chat_template formats messages into the model's expected input string
    # add_generation_prompt=True appends the assistant turn start token
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize the formatted prompt
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_token_count = inputs["input_ids"].shape[1]

    start_time = time.time()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            **gen_config,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,  # Suppress padding warnings
        )

    elapsed = time.time() - start_time

    # Decode only the newly generated tokens (skip the input prompt tokens)
    new_tokens = output_ids[0][input_token_count:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    token_count = len(new_tokens)

    return generated_text, elapsed, token_count


def get_model_info() -> dict:
    """Return basic info about the loaded model for display in the UI."""
    device = _model.device.type if _model is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    return {
        "model_id": MODEL_ID,
        "device": device,
        "loaded": _model is not None,
    }
