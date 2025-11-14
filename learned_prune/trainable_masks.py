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
from utils.schedule import gumbel_sigmoid


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


def _wrap_self_attention(attention_module: nn.Module, mask_param: nn.Parameter, layer_idx: int, model: nn.Module) -> None:
    """Inject trainable head masks into ``BertSelfAttention`` forward pass."""
    if getattr(attention_module, "_trainable_mask_wrapped", False):
        return

    original_forward = attention_module.forward

    def forward(self, *args, **kwargs):
        # Get existing head_mask (if HF passed one)
        incoming_head_mask = kwargs.get("head_mask", None)

        # Build our (per-layer) mask
        if self.training:
            # shape: [1, num_heads, 1, 1]
            base = mask_param[layer_idx].to(
                kwargs.get("hidden_states", args[0]).device,  # infer device from hidden_states
                kwargs.get("hidden_states", args[0]).dtype
            ).view(1, -1, 1, 1)

            temperature = getattr(model, "gumbel_temperature", 1.0)
            use_gumbel  = getattr(model, "use_gumbel", True)
            mask_probs  = gumbel_sigmoid(base, temperature=temperature, training=True, use_gumbel=use_gumbel)

            effective_mask = mask_probs if incoming_head_mask is None else incoming_head_mask * mask_probs
        else:
            effective_mask = incoming_head_mask  # at eval, use provided (typically binary) mask unchanged

        # Inject back and pass everything through unchanged
        kwargs["head_mask"] = effective_mask
        return original_forward(*args, **kwargs)

    attention_module.forward = forward.__get__(attention_module, attention_module.__class__)
    attention_module._trainable_mask_wrapped = True


def _wrap_ffn_intermediate(intermediate_module: nn.Module, mask_param: nn.Parameter, layer_idx: int, model: nn.Module) -> None:
    """Multiply intermediate activations by a trainable mask before passing downstream."""
    if getattr(intermediate_module, "_trainable_mask_wrapped", False):
        return

    original_forward = intermediate_module.forward

    def forward(self, hidden_states, *args, **kwargs):
        if self.training:
            mask = mask_param[layer_idx].to(hidden_states.device, hidden_states.dtype).view(1, 1, -1)
            temperature = getattr(model, "gumbel_temperature", 1.0)
            use_gumbel  = getattr(model, "use_gumbel", True)
            hidden_states = hidden_states * gumbel_sigmoid(mask, temperature=temperature, training=True, use_gumbel=use_gumbel)
        return original_forward(hidden_states, *args, **kwargs)

    intermediate_module.forward = forward.__get__(intermediate_module, intermediate_module.__class__)
    intermediate_module._trainable_mask_wrapped = True


def _wrap_ffn_output(output_module: nn.Module, mask_param: nn.Parameter, layer_idx: int, model: nn.Module) -> None:
    """Apply a trainable mask to the pre-projection FFN activations."""
    if getattr(output_module, "_trainable_mask_wrapped", False):
        return

    original_forward = output_module.forward

    def forward(self, hidden_states, *args, **kwargs):
        if self.training:
            mask = mask_param[layer_idx].to(hidden_states.device, hidden_states.dtype).view(1, 1, -1)
            temperature = getattr(model, "gumbel_temperature", 1.0)
            use_gumbel  = getattr(model, "use_gumbel", True)
            hidden_states = hidden_states * gumbel_sigmoid(mask, temperature=temperature, training=True, use_gumbel=use_gumbel)
        return original_forward(hidden_states, *args, **kwargs)

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

    # Initialize Gumbel temperature and use_gumbel flag on the model
    if not hasattr(model, 'gumbel_temperature'):
        model.gumbel_temperature = 1.0
    if not hasattr(model, 'use_gumbel'):
        model.use_gumbel = True

    if head_mask is not None:
        head_param = _ensure_parameter(model, "trainable_head_mask", head_mask.to(device=device, dtype=dtype))
        for idx, layer in enumerate(get_layers(model)):
            _wrap_self_attention(layer.attention.self, head_param, idx, model)

    if ffn_intermediate_mask is not None:
        ffn_intermediate_param = _ensure_parameter(
            model,
            "trainable_ffn_intermediate_mask",
            ffn_intermediate_mask.to(device=device, dtype=dtype),
        )
        for idx in range(ffn_intermediate_param.shape[0]):
            intermediate_module = get_ffn1(model, idx)
            _wrap_ffn_intermediate(intermediate_module, ffn_intermediate_param, idx, model)

    if ffn_output_mask is not None:
        ffn_output_param = _ensure_parameter(
            model,
            "trainable_ffn_output_mask",
            ffn_output_mask.to(device=device, dtype=dtype),
        )
        for idx in range(ffn_output_param.shape[0]):
            output_module = get_ffn2(model, idx)
            _wrap_ffn_output(output_module, ffn_output_param, idx, model)

    return TrainableMaskHandles(
        head_mask=head_param,
        ffn_intermediate_mask=ffn_intermediate_param,
        ffn_output_mask=ffn_output_param,
    )
