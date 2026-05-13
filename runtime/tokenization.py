from __future__ import annotations

from typing import Literal

from config import TARGET_MODEL_ID
from runtime.deps import require_tokenizer_deps


def load_tokenizer(model_id: str = TARGET_MODEL_ID):
    _, AutoTokenizer = require_tokenizer_deps()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode_prompt(
    tokenizer,
    prompt: str,
    *,
    mode: Literal["plain", "chat"] = "plain",
    enable_thinking: bool = False,
) -> dict[str, "torch.Tensor"]:
    """Tokenize a prompt into tensors ready for model prefill."""

    if mode == "plain":
        return tokenizer(prompt, return_tensors="pt")

    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return tokenizer(text, return_tensors="pt")
