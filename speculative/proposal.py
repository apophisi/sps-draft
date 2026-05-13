from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from runtime.model import ModelRunner, PrefillState
from speculative.sampling import logits_to_probs, sample_token


@dataclass
class DraftProposal:
    """Draft model's k-token proposal and the distributions used to sample it."""

    token_ids: list[int]
    probs: list[np.ndarray]
    state: PrefillState


def propose_k_tokens(
    draft: ModelRunner,
    state: PrefillState,
    *,
    k: int,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> DraftProposal:
    """Use the draft model to autoregressively propose k future tokens.

    The returned `probs[i]` is q_i(.), the draft distribution used to sample
    `token_ids[i]`. SPS verification needs these q_i distributions when
    computing min(1, p_i(x) / q_i(x)).
    """

    if k <= 0:
        raise ValueError("k must be > 0")

    token_ids: list[int] = []
    probs: list[np.ndarray] = []
    current_state = state

    for _ in range(k):
        q = logits_to_probs(
            current_state.next_token_logits,
            temperature=temperature,
            top_k=top_k,
        )
        token_id = sample_token(q, rng)

        token_ids.append(token_id)
        probs.append(q)
        current_state = draft.decode_one(token_id, current_state)

    return DraftProposal(
        token_ids=token_ids,
        probs=probs,
        state=current_state,
    )
