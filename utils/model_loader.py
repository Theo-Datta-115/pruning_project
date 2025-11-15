"""Utilities for loading pretrained and fine-tuned models for pruning experiments."""
from __future__ import annotations

import os
from typing import Tuple

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def load_model_and_tokenizer(
    *,
    model_name: str,
    task_name: str,
    ckpt_dir: str | None,
    use_base_model: bool,
    default_root: str,
) -> Tuple[
    AutoConfig,
    AutoModelForSequenceClassification | AutoModelForQuestionAnswering,
    AutoTokenizer,
    str,
]:
    """Load a task-specific Transformer and its tokenizer.

    Parameters
    ----------
    model_name:
        Base Hugging Face identifier (e.g., ``"bert-base-uncased"``).
    task_name:
        GLUE/SQuAD task name.
    ckpt_dir:
        Optional user-specified checkpoint directory. If ``None`` a default path
        derived from ``default_root`` is used.
    use_base_model:
        When ``True`` the function loads a base (unfine-tuned) checkpoint with the
        appropriate task head if available.
    default_root:
        Root directory holding downloaded checkpoints.
    """

    is_squad = "squad" in task_name
    base_task_type = "question_answering" if is_squad else "sequence_classification"
    model_cls = AutoModelForQuestionAnswering if is_squad else AutoModelForSequenceClassification

    if not use_base_model:
        candidate_dir = ckpt_dir or os.path.join(default_root, model_name, task_name)
        candidate_dir = os.path.abspath(candidate_dir)

        if not os.path.isdir(candidate_dir):
            raise FileNotFoundError(
                f"Checkpoint directory '{candidate_dir}' does not exist. "
                "Download it first or provide `--ckpt_dir`."
            )

        config_path = os.path.join(candidate_dir, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"Checkpoint directory '{candidate_dir}' is missing config.json."
            )

        weight_path = os.path.join(candidate_dir, "model.safetensors")
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(
                f"Checkpoint directory '{candidate_dir}' is missing model.safetensors."
            )

        model_source = candidate_dir
    else:
        base_root = ckpt_dir or os.path.join(default_root, model_name)
        base_root = os.path.abspath(base_root)
        candidate_dir = os.path.join(base_root, f"base_{base_task_type}")
        if os.path.isdir(candidate_dir):
            model_source = candidate_dir
        else:
            model_source = model_name

    config = AutoConfig.from_pretrained(model_source)
    model = model_cls.from_pretrained(model_source, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=None)

    return config, model, tokenizer, model_source


def freeze_model(model, mask_param_ids):
    """Freeze all parameters in the model."""
    for p in model.parameters():
        if id(p) not in mask_param_ids:
            p.requires_grad = False

def unfreeze_layer(model, layer_idx, unfreeze_head=True):
    """Unfreeze one encoder layer (and optionally the classifier head)."""

    layer = model.bert.encoder.layer[layer_idx]
    for p in layer.parameters():
        p.requires_grad = True

    if unfreeze_head and hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True

    print(f"✅ Unfroze layer {layer_idx}" + (" and classifier head" if unfreeze_head else ""))


def unfreeze_model(model):
    """Unfreeze all parameters in the model."""
    for p in model.parameters():
        p.requires_grad = True
    print("✅ Unfroze entire model")