from __future__ import annotations

import numpy as np


def logits_to_probs(
    logits,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> np.ndarray:
    """Convert one-step logits to a NumPy probability distribution.

    `top_k` is optional sampling truncation. It is different from SPS draft
    length k: top_k limits the vocabulary candidates for one sampled token.
    """

    logits_np = as_numpy_1d(logits).astype(np.float64, copy=True)
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")

    logits_np = logits_np / temperature
    if top_k is not None:
        logits_np = apply_top_k(logits_np, top_k)

    logits_np = logits_np - np.max(logits_np)
    exp_logits = np.exp(logits_np)
    total = np.sum(exp_logits)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("invalid logits: cannot build a probability distribution")
    return exp_logits / total


def sample_token(probs: np.ndarray, rng: np.random.Generator) -> int:
    probs = normalize(probs)
    return int(rng.choice(len(probs), p=probs))


def sample_residual(
    target_probs: np.ndarray,
    draft_probs: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Sample from norm(max(target_probs - draft_probs, 0))."""

    residual = np.maximum(
        np.asarray(target_probs, dtype=np.float64)
        - np.asarray(draft_probs, dtype=np.float64),
        0.0,
    )
    if np.sum(residual) <= 0.0:
        return sample_token(target_probs, rng)
    return sample_token(residual, rng)


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = np.sum(values)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("cannot normalize an empty or all-zero distribution")
    return values / total


def apply_top_k(logits: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if top_k >= logits.shape[-1]:
        return logits

    filtered = np.full_like(logits, -np.inf)
    top_indices = np.argpartition(logits, -top_k)[-top_k:]
    filtered[top_indices] = logits[top_indices]
    return filtered


def as_numpy_1d(tensor_or_array) -> np.ndarray:
    if hasattr(tensor_or_array, "detach"):
        tensor_or_array = tensor_or_array.detach().float().cpu().numpy()
    array = np.asarray(tensor_or_array)
    return np.squeeze(array)
