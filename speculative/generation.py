from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from runtime.model import ModelRunner, PrefillState
from speculative.proposal import DraftProposal, propose_k_tokens
from speculative.verification import VerificationResult, verify_k_tokens


@dataclass
class GenerationStats:
    rounds: int = 0
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    acceptance_lengths: list[int] = field(default_factory=list)

    @property
    def avg_accept(self) -> float:
        if not self.acceptance_lengths:
            return 0.0
        return sum(self.acceptance_lengths) / len(self.acceptance_lengths)

    @property
    def accept_rate(self) -> float:
        if self.proposed_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.proposed_tokens


@dataclass
class GenerationResult:
    token_ids: list[int]
    draft_state: PrefillState
    target_state: PrefillState
    stats: GenerationStats
    stopped_by_eos: bool
    rounds: list[VerificationResult]


def speculative_generate(
    draft: ModelRunner,
    target: ModelRunner,
    draft_state: PrefillState,
    target_state: PrefillState,
    *,
    max_new_tokens: int,
    draft_steps: int,
    rng: np.random.Generator,
    eos_token_id: int | None = None,
    temperature: float = 0.0,
    top_k: int | None = None,
) -> GenerationResult:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    if draft_steps <= 0:
        raise ValueError("draft_steps must be > 0")

    generated_token_ids: list[int] = []
    rounds: list[VerificationResult] = []
    stats = GenerationStats()
    stopped_by_eos = False
    current_draft_state = draft_state
    current_target_state = target_state

    while len(generated_token_ids) < max_new_tokens:
        remaining = max_new_tokens - len(generated_token_ids)
        k = min(draft_steps, remaining)
        sample_bonus = remaining > k

        proposal = propose_k_tokens(
            draft,
            current_draft_state,
            k=k,
            rng=rng,
            temperature=temperature,
            top_k=top_k,
        )
        verification = verify_k_tokens(
            target,
            current_target_state,
            proposal,
            rng=rng,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_token_id,
            sample_bonus=sample_bonus,
        )

        round_token_ids = verification.token_ids[:remaining]
        generated_token_ids.extend(round_token_ids)
        current_target_state = verification.state
        current_draft_state = advance_draft_state_after_round(
            draft,
            current_draft_state,
            proposal,
            verification,
        )

        stats.rounds += 1
        stats.proposed_tokens += verification.proposed_count
        stats.accepted_tokens += verification.accepted_count
        stats.acceptance_lengths.append(verification.acceptance_length)
        rounds.append(verification)

        if eos_token_id is not None and eos_token_id in round_token_ids:
            stopped_by_eos = True
            break

    return GenerationResult(
        token_ids=generated_token_ids,
        draft_state=current_draft_state,
        target_state=current_target_state,
        stats=stats,
        stopped_by_eos=stopped_by_eos,
        rounds=rounds,
    )


def advance_state(
    runner: ModelRunner,
    state: PrefillState,
    token_ids: list[int],
) -> PrefillState:
    return runner.decode_many(token_ids, state)


def advance_draft_state_after_round(
    runner: ModelRunner,
    state: PrefillState,
    proposal: DraftProposal,
    verification: VerificationResult,
) -> PrefillState:
    if verification.accepted_all:
        current_state = proposal.state
        if verification.bonus_token_id is not None:
            current_state = runner.decode_one(verification.bonus_token_id, current_state)
        return current_state
    return runner.decode_many(verification.token_ids, state)
