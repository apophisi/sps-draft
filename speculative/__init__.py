from __future__ import annotations

from speculative.generation import (
    GenerationResult,
    GenerationStats,
    speculative_generate,
)
from speculative.proposal import DraftProposal, propose_k_tokens
from speculative.sampling import logits_to_probs, sample_residual, sample_token
from speculative.verification import (
    VerificationResult,
    acceptance_probability,
    verify_k_tokens,
)


__all__ = [
    "DraftProposal",
    "GenerationResult",
    "GenerationStats",
    "VerificationResult",
    "acceptance_probability",
    "logits_to_probs",
    "propose_k_tokens",
    "sample_residual",
    "sample_token",
    "speculative_generate",
    "verify_k_tokens",
]
