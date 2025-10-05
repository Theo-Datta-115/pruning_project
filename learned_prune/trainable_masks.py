"""Utilities for promoting pruning masks into trainable model parameters.

This module exposes helpers that convert static pruning masks into `nn.Parameter`
objects and seamlessly integrate them into a Hugging Face Transformer forward
pass. The masks become part of the computational graph, making it possible to
optimise them with standard gradient-based methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from utils.arch import get_layers, get_ffn1, get_ffn2


@dataclass
class TrainableMaskHandles:
    """Bookkeeping container returned by :func:`make_masks_trainable`.

    Attributes
    ----------
    head_mask : torch.nn.Parameter | None
        Trainable head mask registered on the model (``None`` if not supplied).
    ffn_intermediate_mask : torch.nn.Parameter | None
        Trainable mask applied to the intermediate feed-forward activations.
    ffn_output_mask : torch.nn.Parameter | None
        Trainable mask applied to the feed-forward output activations.
    """

    head_mask: Optional[nn.Parameter]
    ffn_intermediate_mask: Optional[nn.Parameter]
    ffn_output_mask: Optional[nn.Parameter]


def _ensure_parameter(module: nn.Module, name: str, value: torch.Tensor) -> nn.Parameter:
    """Register a tensor as an ``nn.Parameter`` on ``module``.

    If a parameter with ``name`` already exists, it will be overwritten with the
    provided value to keep the interface idempotent.
    """

    if hasattr(module, name):
        delattr(module, name)
    param = nn.Parameter(value)
    module.register_parameter(name, param)
    return param


def _wrap_self_attention(attention_module: nn.Module, mask_param: nn.Parameter, layer_idx: int) -> None:
    """Inject trainable head masks into ``BertSelfAttention`` forward pass."""
    if getattr(attention_module, "_trainable_mask_wrapped", False):
        return

    original_forward = attention_module.forward

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mask = mask_param[layer_idx].to(hidden_states.device, hidden_states.dtype).view(1, -1, 1, 1)
        effective_mask = mask if head_mask is None else head_mask * torch.sigmoid(mask)
        return original_forward(
            hidden_states,
            attention_mask,
            effective_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )

    attention_module.forward = forward.__get__(attention_module, attention_module.__class__)
    attention_module._trainable_mask_wrapped = True


def _wrap_ffn_intermediate(intermediate_module: nn.Module, mask_param: nn.Parameter, layer_idx: int) -> None:
    """Multiply intermediate activations by a trainable mask before passing downstream."""
    if getattr(intermediate_module, "_trainable_mask_wrapped", False):
        return

    original_forward = intermediate_module.forward

    def forward(self, hidden_states):
        mask = mask_param[layer_idx].to(hidden_states.device, hidden_states.dtype).view(1, 1, -1)
        hidden_states = hidden_states * torch.sigmoid(mask)
        return original_forward(hidden_states)

    intermediate_module.forward = forward.__get__(intermediate_module, intermediate_module.__class__)
    intermediate_module._trainable_mask_wrapped = True


def _wrap_ffn_output(output_module: nn.Module, mask_param: nn.Parameter, layer_idx: int) -> None:
    """Apply a trainable mask to the pre-projection FFN activations."""
    if getattr(output_module, "_trainable_mask_wrapped", False):
        return

    original_forward = output_module.forward

    def forward(self, hidden_states, input_tensor):
        mask = mask_param[layer_idx].to(hidden_states.device, hidden_states.dtype).view(1, 1, -1)
        hidden_states = hidden_states * torch.sigmoid(mask)
        return original_forward(hidden_states, input_tensor)

    output_module.forward = forward.__get__(output_module, output_module.__class__)
    output_module._trainable_mask_wrapped = True


def make_masks_trainable(
    model: nn.Module,
    *,
    head_mask: Optional[torch.Tensor] = None,
    ffn_intermediate_mask: Optional[torch.Tensor] = None,
    ffn_output_mask: Optional[torch.Tensor] = None,
) -> TrainableMaskHandles:
    """Promote binary masks to trainable parameters and integrate them into ``model``.

    Parameters
    ----------
    model:
        The Transformer model (e.g., ``AutoModelForSequenceClassification``) to augment.
    head_mask:
        Optional tensor of shape ``[num_layers, num_heads]``. When provided, each layer's
        attention head outputs are modulated by a learnable parameter vector.
    ffn_intermediate_mask:
        Optional tensor of shape ``[num_layers, hidden_size]``. Each element scales the
        post-activation hidden representation before entering the intermediate linear.
    ffn_output_mask:
        Optional tensor of shape ``[num_layers, intermediate_size]`` used to scale the
        intermediate activations prior to the FFN output projection.

    Returns
    -------
    TrainableMaskHandles
        A container exposing the newly registered mask parameters.
    """

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    head_param: Optional[nn.Parameter] = None
    ffn_intermediate_param: Optional[nn.Parameter] = None
    ffn_output_param: Optional[nn.Parameter] = None

    if head_mask is not None:
        head_param = _ensure_parameter(model, "trainable_head_mask", head_mask.to(device=device, dtype=dtype))
        for idx, layer in enumerate(get_layers(model)):
            _wrap_self_attention(layer.attention.self, head_param, idx)

    if ffn_intermediate_mask is not None:
        ffn_intermediate_param = _ensure_parameter(
            model,
            "trainable_ffn_intermediate_mask",
            ffn_intermediate_mask.to(device=device, dtype=dtype),
        )
        for idx in range(ffn_intermediate_param.shape[0]):
            intermediate_module = get_ffn1(model, idx)
            _wrap_ffn_intermediate(intermediate_module, ffn_intermediate_param, idx)

    if ffn_output_mask is not None:
        ffn_output_param = _ensure_parameter(
            model,
            "trainable_ffn_output_mask",
            ffn_output_mask.to(device=device, dtype=dtype),
        )
        for idx in range(ffn_output_param.shape[0]):
            output_module = get_ffn2(model, idx)
            _wrap_ffn_output(output_module, ffn_output_param, idx)

    return TrainableMaskHandles(
        head_mask=head_param,
        ffn_intermediate_mask=ffn_intermediate_param,
        ffn_output_mask=ffn_output_param,
    )
