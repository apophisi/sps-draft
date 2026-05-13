from __future__ import annotations


INSTALL_HINT = "uv add torch transformers accelerate"


def require_model_deps():
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise ImportError(
            "This module needs PyTorch and Transformers. Install them with:\n"
            f"  {INSTALL_HINT}"
        ) from error
    return torch, AutoModelForCausalLM


def require_tokenizer_deps():
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "This module needs PyTorch and Transformers. Install them with:\n"
            f"  {INSTALL_HINT}"
        ) from error
    return torch, AutoTokenizer


def resolve_device(torch, device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(torch, dtype: str, device):
    if dtype == "auto":
        if device.type == "cuda":
            return torch.bfloat16
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype}")
