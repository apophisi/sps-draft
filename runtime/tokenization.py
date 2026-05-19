from __future__ import annotations

from typing import Literal

from config import TARGET_MODEL_ID
from runtime.deps import configure_hf_endpoint, require_tokenizer_deps


def load_tokenizer(
    model_id: str = TARGET_MODEL_ID,
    *,
    hf_endpoint: str | None = None,
    local_files_only: bool = False,
):
    endpoint = configure_hf_endpoint(hf_endpoint)
    _, AutoTokenizer = require_tokenizer_deps()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
    except OSError as error:
        raise RuntimeError(
            f"Failed to load tokenizer for {model_id!r} from {endpoint!r}.\n"
            "Try another endpoint, for example:\n"
            "  uv run python main.py --hf-endpoint https://huggingface.co\n"
            "Or pre-download the model and run with --local-files-only."
        ) from error
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
