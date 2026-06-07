from __future__ import annotations

import torch


def logits_to_probs(
    logits,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Convert one-step logits to a Torch probability distribution."""

    logits_t = as_torch_1d(logits).float()
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")

    logits_t = logits_t / temperature
    if top_k is not None:
        logits_t = apply_top_k(logits_t, top_k)

    probs = torch.softmax(logits_t, dim=-1)
    total = probs.sum()
    if not torch.isfinite(total).item() or float(total.item()) <= 0.0:
        raise ValueError("invalid logits: cannot build a probability distribution")
    return probs


def sample_token(probs: torch.Tensor, rng=None) -> int:
    probs = normalize(probs)
    generator = make_torch_generator(probs, rng)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def sample_residual(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    rng=None,
) -> int:
    """Sample from norm(max(target_probs - draft_probs, 0))."""

    residual = torch.clamp(
        as_torch_1d(target_probs) - as_torch_1d(draft_probs),
        min=0.0,
    )
    if float(residual.sum().item()) <= 0.0:
        return sample_token(target_probs, rng)
    return sample_token(residual, rng)


def normalize(values: torch.Tensor) -> torch.Tensor:
    values = as_torch_1d(values).float()
    total = values.sum()
    if not torch.isfinite(total).item() or float(total.item()) <= 0.0:
        raise ValueError("cannot normalize an empty or all-zero distribution")
    return values / total


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if top_k >= logits.shape[-1]:
        return logits

    top_values, top_indices = torch.topk(logits, k=top_k, dim=-1)
    filtered = torch.full_like(logits, -torch.inf)
    return filtered.scatter(dim=-1, index=top_indices, src=top_values)


def as_torch_1d(tensor_or_array) -> torch.Tensor:
    if torch.is_tensor(tensor_or_array):
        return tensor_or_array.detach().squeeze()
    return torch.as_tensor(tensor_or_array).squeeze()


def make_torch_generator(probs: torch.Tensor, rng) -> torch.Generator | None:
    if rng is None or not hasattr(rng, "integers"):
        return None
    generator = torch.Generator(device=probs.device)
    seed = int(rng.integers(0, 2**63 - 1))
    generator.manual_seed(seed)
    return generator
