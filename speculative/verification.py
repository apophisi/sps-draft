from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from runtime.model import ModelRunner, PrefillState
from speculative.proposal import DraftProposal
from speculative.sampling import logits_to_probs, sample_residual, sample_token


@dataclass
class VerificationResult:
    """Result of checking draft tokens against the target model."""

    accepted_token_ids: list[int]
    corrected_token_id: int | None
    bonus_token_id: int | None
    proposed_count: int
    target_probs: list[np.ndarray]
    state: PrefillState

    @property
    def token_ids(self) -> list[int]:
        token_ids = self.accepted_token_ids[:]
        if self.corrected_token_id is not None:
            token_ids.append(self.corrected_token_id)
        if self.bonus_token_id is not None:
            token_ids.append(self.bonus_token_id)
        return token_ids

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_token_ids)

    @property
    def acceptance_length(self) -> int:
        return self.accepted_count

    @property
    def accept_rate(self) -> float:
        if self.proposed_count == 0:
            return 0.0
        return self.accepted_count / self.proposed_count

    @property
    def rejected(self) -> bool:
        return self.corrected_token_id is not None

    @property
    def accepted_all(self) -> bool:
        return self.corrected_token_id is None


def verify_k_tokens(
    target: ModelRunner,
    state: PrefillState,
    proposal: DraftProposal,
    *,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
    sample_bonus: bool = True,
) -> VerificationResult:
    """Verify a draft proposal with the target model.

    For proposed token x_i, q_i is the draft distribution saved in the proposal
    and p_i is the target distribution at the same prefix. The token is accepted
    with probability min(1, p_i(x_i) / q_i(x_i)).

    On the first rejection, this function samples one corrected token from the
    residual distribution norm(max(p_i - q_i, 0)) and stops the round. If every
    proposed token is accepted, it samples one bonus token from the target model.
    """

    if len(proposal.token_ids) != len(proposal.probs):
        raise ValueError("proposal token_ids and probs must have the same length")

    proposed_count = len(proposal.token_ids)
    accepted_token_ids: list[int] = []
    target_probs: list[np.ndarray] = []
    current_state = state

    for token_id, draft_probs in zip(proposal.token_ids, proposal.probs):
        target_distribution = logits_to_probs(
            current_state.next_token_logits,
            temperature=temperature,
            top_k=top_k,
        )
        target_probs.append(target_distribution)

        accept_prob = acceptance_probability(
            token_id=token_id,
            target_probs=target_distribution,
            draft_probs=draft_probs,
        )

        if rng.random() <= accept_prob:
            accepted_token_ids.append(token_id)
            current_state = target.decode_one(token_id, current_state)
            if token_id == eos_token_id:
                return VerificationResult(
                    accepted_token_ids=accepted_token_ids,
                    corrected_token_id=None,
                    bonus_token_id=None,
                    proposed_count=proposed_count,
                    target_probs=target_probs,
                    state=current_state,
                )
            continue

        corrected_token_id = sample_residual(
            target_distribution,
            draft_probs,
            rng,
        )
        current_state = target.decode_one(corrected_token_id, current_state)
        return VerificationResult(
            accepted_token_ids=accepted_token_ids,
            corrected_token_id=corrected_token_id,
            bonus_token_id=None,
            proposed_count=proposed_count,
            target_probs=target_probs,
            state=current_state,
        )

    if not sample_bonus:
        return VerificationResult(
            accepted_token_ids=accepted_token_ids,
            corrected_token_id=None,
            bonus_token_id=None,
            proposed_count=proposed_count,
            target_probs=target_probs,
            state=current_state,
        )

    bonus_distribution = logits_to_probs(
        current_state.next_token_logits,
        temperature=temperature,
        top_k=top_k,
    )
    bonus_token_id = sample_token(bonus_distribution, rng)
    current_state = target.decode_one(bonus_token_id, current_state)

    return VerificationResult(
        accepted_token_ids=accepted_token_ids,
        corrected_token_id=None,
        bonus_token_id=bonus_token_id,
        proposed_count=proposed_count,
        target_probs=target_probs,
        state=current_state,
    )


def acceptance_probability(
    *,
    token_id: int,
    target_probs: np.ndarray,
    draft_probs: np.ndarray,
) -> float:
    p = float(target_probs[token_id])
    q = float(draft_probs[token_id])
    if q <= 0.0:
        return 1.0 if p > 0.0 else 0.0
    return min(1.0, p / q)
