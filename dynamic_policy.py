"""Dynamic draft depth policies for speculative decoding (Proposal Part 2).

Extends the fixed-K framework by varying how many draft tokens are proposed each
round. Strategies:
  - p_max early stop: stop when p_max falls below threshold.
  - top-1 / top-2 margin: stop when (p_top1 - p_top2) falls below margin.

Per the proposal: K_max = 8; after each drafted token, check the stopping rule.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from runtime.model import ModelRunner, PrefillState
from speculative.generation import GenerationStats, advance_state
from speculative.proposal import DraftProposal
from speculative.sampling import logits_to_probs, sample_token
from speculative.verification import VerificationResult, verify_k_tokens


@dataclass
class DynamicGenerationStats(GenerationStats):
    """Stats for dynamic draft depth, including average draft length."""

    draft_lengths: list[int] = field(default_factory=list)

    @property
    def average_draft_length(self) -> float:
        if not self.draft_lengths:
            return 0.0
        return sum(self.draft_lengths) / len(self.draft_lengths)


@dataclass
class DynamicGenerationResult:
    token_ids: list[int]
    draft_state: PrefillState
    target_state: PrefillState
    stats: DynamicGenerationStats
    stopped_by_eos: bool
    rounds: list[VerificationResult]


class DraftDepthPolicy(ABC):
    """Decide whether to stop proposing more draft tokens this round."""

    @property
    @abstractmethod
    def K_max(self) -> int:
        """Maximum speculative draft depth per round (Proposal: K_max)."""


@dataclass(frozen=True)
class FixedDepthPolicy(DraftDepthPolicy):
    """Propose exactly K tokens per round (no early stop)."""

    K: int

    @property
    def K_max(self) -> int:
        return self.K

    def should_stop_after_token(
        self,
        probs: np.ndarray,
        *,
        tokens_proposed: int,
    ) -> bool:
        return tokens_proposed >= self.K

    @property
    def strategy_name(self) -> str:
        return f"K={self.K}"


@dataclass(frozen=True)
class PMaxEarlyStopPolicy(DraftDepthPolicy):
    """Stop when p_max < threshold after a drafted token (Proposal 3.2)."""

    threshold: float
    K_max: int = 8

    def should_stop_after_token(
        self,
        probs: np.ndarray,
        *,
        tokens_proposed: int,
    ) -> bool:
        p_max = float(np.max(probs))
        return p_max < self.threshold

    @property
    def strategy_name(self) -> str:
        return f"p_max > {self.threshold}"


@dataclass(frozen=True)
class Top1Top2MarginEarlyStopPolicy(DraftDepthPolicy):
    """Stop when (p_top1 - p_top2) < margin after a drafted token."""

    margin: float
    K_max: int = 8

    def should_stop_after_token(
        self,
        probs: np.ndarray,
        *,
        tokens_proposed: int,
    ) -> bool:
        if probs.size < 2:
            return True
        top2 = np.partition(probs, -2)[-2:]
        p_top1, p_top2 = float(np.max(top2)), float(np.min(top2))
        return (p_top1 - p_top2) < self.margin

    @property
    def strategy_name(self) -> str:
        return f"top1-top2 margin > {self.margin}"


def select_token(
    logits,
    *,
    rng: np.random.Generator,
    temperature: float = 0.0,
    top_k: int | None = None,
) -> tuple[int, np.ndarray]:
    """Greedy (temperature=0) or sampled token plus draft distribution q_i.

    Stopping rules use softmax at temperature=1.0 so p_max reflects model
    confidence (Proposal: compute p_max from the current prediction distribution).
    """

    policy_temperature = 1.0 if temperature <= 0.0 else temperature
    probs = logits_to_probs(logits, temperature=policy_temperature, top_k=top_k)
    if temperature <= 0.0:
        token_id = int(np.argmax(probs))
    else:
        token_id = sample_token(probs, rng)
    return token_id, probs


def propose_dynamic_tokens(
    draft: ModelRunner,
    state: PrefillState,
    policy: DraftDepthPolicy,
    *,
    rng: np.random.Generator,
    temperature: float = 0.0,
    top_k: int | None = None,
    max_new_tokens: int | None = None,
) -> DraftProposal:
    """Propose draft tokens until the policy stops or K_max is reached."""

    K_limit = policy.K_max
    if max_new_tokens is not None:
        K_limit = min(K_limit, max_new_tokens)
    if K_limit <= 0:
        return DraftProposal(token_ids=[], probs=[], state=state)

    token_ids: list[int] = []
    probs_list: list[np.ndarray] = []
    current_state = state

    for _ in range(K_limit):
        token_id, probs = select_token(
            current_state.next_token_logits,
            rng=rng,
            temperature=temperature,
            top_k=top_k,
        )
        token_ids.append(token_id)
        probs_list.append(probs)
        current_state = draft.decode_one(token_id, current_state)

        if policy.should_stop_after_token(
            probs,
            tokens_proposed=len(token_ids),
        ):
            break

    return DraftProposal(
        token_ids=token_ids,
        probs=probs_list,
        state=current_state,
    )


def speculative_generate_dynamic(
    draft: ModelRunner,
    target: ModelRunner,
    draft_state: PrefillState,
    target_state: PrefillState,
    policy: DraftDepthPolicy,
    *,
    max_new_tokens: int,
    rng: np.random.Generator,
    eos_token_id: int | None = None,
    temperature: float = 0.0,
    top_k: int | None = None,
) -> DynamicGenerationResult:
    """Speculative decoding loop with dynamic draft depth per round."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")

    generated_token_ids: list[int] = []
    rounds: list[VerificationResult] = []
    stats = DynamicGenerationStats()
    stopped_by_eos = False
    current_draft_state = draft_state
    current_target_state = target_state

    while len(generated_token_ids) < max_new_tokens:
        remaining = max_new_tokens - len(generated_token_ids)

        proposal = propose_dynamic_tokens(
            draft,
            current_draft_state,
            policy,
            rng=rng,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=remaining,
        )
        draft_length = len(proposal.token_ids)
        stats.draft_lengths.append(draft_length)

        if draft_length == 0:
            break

        sample_bonus = remaining > draft_length
        verify_temperature = 1.0 if temperature <= 0.0 else temperature
        verification = verify_k_tokens(
            target,
            current_target_state,
            proposal,
            rng=rng,
            temperature=verify_temperature,
            top_k=top_k,
            eos_token_id=eos_token_id,
            sample_bonus=sample_bonus,
        )

        round_token_ids = verification.token_ids[:remaining]
        generated_token_ids.extend(round_token_ids)
        current_target_state = verification.state
        current_draft_state = advance_state(draft, current_draft_state, round_token_ids)

        stats.rounds += 1
        stats.proposed_tokens += verification.proposed_count
        stats.accepted_tokens += verification.accepted_count
        stats.acceptance_lengths.append(verification.acceptance_length)
        rounds.append(verification)

        if eos_token_id is not None and eos_token_id in round_token_ids:
            stopped_by_eos = True
            break

    return DynamicGenerationResult(
        token_ids=generated_token_ids,
        draft_state=current_draft_state,
        target_state=current_target_state,
        stats=stats,
        stopped_by_eos=stopped_by_eos,
        rounds=rounds,
    )
