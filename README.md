# SPS Draft

Minimal speculative decoding playground using `Qwen/Qwen3-0.6B` as the draft
model and `Qwen/Qwen3-1.7B` as the target model.

The project is currently organized as a lightweight experiment repo: modules
live at the repository root, with separate folders for runtime model execution,
pipeline orchestration, and speculative decoding logic.

## Layout

```text
config.py                 # model ids and shared constants
main.py                   # smoke-test entrypoint

runtime/
  deps.py                 # torch/transformers dependency checks, device, dtype
  model.py                # ModelRunner, PrefillState, prefill, cached decode
  tokenization.py         # tokenizer loading and prompt/chat-template encoding

pipeline/
  prefill.py              # draft/target loading and prompt prefill orchestration

speculative/
  sampling.py             # NumPy logits -> probs, top-k filtering, token sampling
  proposal.py             # draft model proposes k speculative tokens
```

## Setup

```bash
uv add torch transformers accelerate numpy
```

If you use a CUDA-specific PyTorch wheel, install `torch` from the matching
PyTorch index first, then add the remaining packages.

Model downloads use the Hugging Face mirror by default:

```text
HF_ENDPOINT=https://hf-mirror.com
```

To override it for one run:

```bash
uv run python main.py --hf-endpoint https://huggingface.co
```

Hugging Face files are cached under `~/.cache/huggingface/hub` unless you set
`HF_HOME` or `HUGGINGFACE_HUB_CACHE`.

If the models are already cached locally, skip network access with:

```bash
uv run python main.py --local-files-only
```

## Current Flow

1. `pipeline.load_draft_and_target()` loads tokenizer, draft model, and target
   model.
2. `runtime.encode_prompt()` tokenizes a plain prompt or Qwen chat prompt.
3. `pipeline.prefill_both()` runs the prompt through both models and returns
   their KV-cache states.
4. `speculative.propose_k_tokens()` uses the draft model cache to sample `k`
   future tokens with NumPy sampling.
5. `speculative.verify_k_tokens()` verifies the proposal with the target model.
6. `speculative.speculative_generate()` repeats proposal and verification until
   `max_new_tokens` or EOS.

Example:

```python
import numpy as np

from pipeline import load_draft_and_target, prefill_both
from runtime import encode_prompt
from speculative import speculative_generate


tokenizer, draft, target = load_draft_and_target()
batch = encode_prompt(
    tokenizer,
    "请用一句话解释 speculative decoding。",
    mode="chat",
    enable_thinking=False,
)
draft_state, target_state = prefill_both(draft, target, batch)

generation = speculative_generate(
    draft,
    target,
    draft_state,
    target_state,
    max_new_tokens=128,
    draft_steps=4,
    rng=np.random.default_rng(0),
    temperature=1.0,
)

print(tokenizer.decode(generation.token_ids, skip_special_tokens=True))
print(generation.stats.avg_accept)
print(generation.stats.accept_rate)
```

Run generation:

```bash
uv run python main.py -k 4 --max-new-tokens 128
```

## Notes

- SPS draft length `k` means how many future tokens the draft model proposes in
  one speculative round.
- The optional `top_k` argument in `speculative.sampling.logits_to_probs()` is a
  per-step vocabulary truncation setting. It is different from SPS draft length
  `k`.
- `avg_accept` is the average accepted draft length per speculative round.
- `accept rate` is total accepted draft tokens divided by total proposed draft
  tokens.
