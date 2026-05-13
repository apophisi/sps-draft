from __future__ import annotations

from dataclasses import dataclass

from runtime.deps import require_model_deps, resolve_device, resolve_dtype


@dataclass
class PrefillState:
    """KV-cache state produced by feeding the full prompt once."""

    input_ids: "torch.Tensor"
    attention_mask: "torch.Tensor"
    past_key_values: object
    logits: "torch.Tensor"

    @property
    def next_token_logits(self) -> "torch.Tensor":
        return self.logits[:, -1, :]


def no_grad(method):
    def wrapper(self, *args, **kwargs):
        with self.torch.no_grad():
            return method(self, *args, **kwargs)

    return wrapper


class ModelRunner:
    """Thin causal-LM wrapper for prefill and cached one-token decode."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
        device_map: str | None = None,
    ) -> None:
        torch, AutoModelForCausalLM = require_model_deps()

        self.torch = torch
        self.model_id = model_id
        self.device = resolve_device(torch, device)
        self.dtype = resolve_dtype(torch, dtype, self.device)

        load_kwargs = {
            "torch_dtype": self.dtype,
            "trust_remote_code": True,
        }
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self.model.eval()

        if device_map is None:
            self.model.to(self.device)

    @property
    def model_device(self) -> "torch.device":
        return next(self.model.parameters()).device

    @no_grad
    def prefill(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor | None" = None,
    ) -> PrefillState:
        input_ids = input_ids.to(self.model_device)
        if attention_mask is None:
            attention_mask = self.torch.ones_like(input_ids, device=self.model_device)
        else:
            attention_mask = attention_mask.to(self.model_device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        return PrefillState(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=outputs.past_key_values,
            logits=outputs.logits,
        )

    @no_grad
    def decode_one(self, token_id: int, state: PrefillState) -> PrefillState:
        next_input_ids = self.torch.tensor(
            [[token_id]],
            dtype=state.input_ids.dtype,
            device=self.model_device,
        )
        next_attention = self.torch.ones(
            (state.attention_mask.shape[0], 1),
            dtype=state.attention_mask.dtype,
            device=self.model_device,
        )
        attention_mask = self.torch.cat([state.attention_mask, next_attention], dim=1)

        outputs = self.model(
            input_ids=next_input_ids,
            attention_mask=attention_mask,
            past_key_values=state.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        input_ids = self.torch.cat([state.input_ids, next_input_ids], dim=1)

        return PrefillState(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=outputs.past_key_values,
            logits=outputs.logits,
        )
