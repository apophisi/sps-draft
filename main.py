from __future__ import annotations

from pipeline import load_draft_and_target, prefill_both
from runtime import encode_prompt


def main() -> None:
    tokenizer, draft, target = load_draft_and_target()
    batch = encode_prompt(
        tokenizer,
        "请用一句话解释 speculative decoding。",
        mode="chat",
        enable_thinking=False,
    )
    draft_state, target_state = prefill_both(draft, target, batch)

    print("input shape:", tuple(batch["input_ids"].shape))
    print("draft next logits:", tuple(draft_state.next_token_logits.shape))
    print("target next logits:", tuple(target_state.next_token_logits.shape))
    print("draft cache type:", type(draft_state.past_key_values).__name__)
    print("target cache type:", type(target_state.past_key_values).__name__)


if __name__ == "__main__":
    main()
