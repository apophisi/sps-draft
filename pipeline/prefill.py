from __future__ import annotations

from config import DRAFT_MODEL_ID, TARGET_MODEL_ID
from runtime.model import ModelRunner, PrefillState
from runtime.tokenization import load_tokenizer


def load_draft_and_target(
    *,
    device: str = "auto",
    dtype: str = "auto",
    device_map: str | None = None,
) -> tuple[object, ModelRunner, ModelRunner]:
    """Load tokenizer, Qwen3-0.6B draft model, and Qwen3-1.7B target model."""

    tokenizer = load_tokenizer(TARGET_MODEL_ID)
    draft = ModelRunner(
        DRAFT_MODEL_ID,
        device=device,
        dtype=dtype,
        device_map=device_map,
    )
    target = ModelRunner(
        TARGET_MODEL_ID,
        device=device,
        dtype=dtype,
        device_map=device_map,
    )
    return tokenizer, draft, target


def prefill_both(
    draft: ModelRunner,
    target: ModelRunner,
    batch: dict[str, "torch.Tensor"],
) -> tuple[PrefillState, PrefillState]:
    """Run the same prompt through draft and target, returning both KV caches."""

    draft_state = draft.prefill(
        batch["input_ids"],
        batch.get("attention_mask"),
    )
    target_state = target.prefill(
        batch["input_ids"],
        batch.get("attention_mask"),
    )
    return draft_state, target_state
