from __future__ import annotations

import argparse

import numpy as np

from pipeline import load_draft_and_target, prefill_both
from runtime import encode_prompt
from speculative import speculative_generate


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    tokenizer, draft, target = load_draft_and_target(
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
        hf_endpoint=args.hf_endpoint,
        local_files_only=args.local_files_only,
    )
    batch = encode_prompt(
        tokenizer,
        args.prompt,
        mode=args.prompt_mode,
        enable_thinking=args.enable_thinking,
    )
    draft_state, target_state = prefill_both(draft, target, batch)

    generation = speculative_generate(
        draft,
        target,
        draft_state,
        target_state,
        max_new_tokens=args.max_new_tokens,
        draft_steps=args.draft_steps,
        rng=rng,
        eos_token_id=tokenizer.eos_token_id,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    generated_text = tokenizer.decode(
        generation.token_ids,
        skip_special_tokens=True,
    )

    print("prompt:")
    print(args.prompt)
    print()
    print("answer:")
    print(generated_text)
    print()
    print("stats:")
    print("  input shape:       ", tuple(batch["input_ids"].shape))
    print("  generated tokens:  ", len(generation.token_ids))
    print("  rounds:            ", generation.stats.rounds)
    print("  avg_accept:        ", f"{generation.stats.avg_accept:.3f}")
    print("  accept rate:       ", f"{generation.stats.accept_rate:.3f}")
    print("  stopped by eos:    ", generation.stopped_by_eos)
    print("  target cache type: ", type(generation.target_state.past_key_values).__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default="请用一句话解释 speculative decoding。",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["plain", "chat"],
        default="chat",
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("-k", "--draft-steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
