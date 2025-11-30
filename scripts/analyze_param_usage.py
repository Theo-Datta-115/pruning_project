#!/usr/bin/env python3
"""Summarise how parameters are distributed across a saved Transformer model.

This helper script loads a fine-tuned checkpoint (e.g., BERT or DistilBERT) and
reports how many parameters belong to major architectural components such as the
embedding tables, encoder stack, pooler, and task head. Optionally, the encoder
breakdown can include a per-layer view so you can see which layers dominate the
parameter budget. When encoder details are requested, each layer is further
divided into attention (self- and output projections), feed-forward (intermediate
and output) and LayerNorm parameters to highlight what portion of the encoder is
allocated to each sub-module.

python analyze_param_usage.py /n/netscratch/sham_lab/Everyone/tdatta/pruning/outputs/layerwise_9_sq 
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        help="Path to the saved model directory or a Hugging Face model identifier.",
    )
    parser.add_argument(
        "--class-name",
        default=None,
        help=(
            "Optional Transformers class name to instantiate (e.g. "
            "'AutoModelForSequenceClassification' or 'AutoModelForQuestionAnswering'). "
            "When omitted, the script will try to infer a suitable class from the "
            "checkpoint metadata."
        ),
    )
    parser.add_argument(
        "--trainable-only",
        action="store_true",
        help="Count only parameters with requires_grad=True.",
    )
    parser.add_argument(
        "--show-submodules",
        action="store_true",
        help=(
            "Include per-layer statistics for encoder/transformer blocks in the output."
        ),
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="Number of decimal places to display for percentages (default: 2).",
    )
    return parser.parse_args()


def infer_model_class(config: AutoConfig, explicit_class: str | None):
    """Return a transformers.PreTrainedModel subclass to instantiate."""
    if explicit_class is not None:
        cls = getattr(transformers, explicit_class, None)
        if cls is None:
            raise ValueError(
                f"Unknown class '{explicit_class}'. Ensure it is available in transformers."
            )
        return cls

    architectures = getattr(config, "architectures", None) or []
    for arch in architectures:
        cls = getattr(transformers, arch, None)
        if cls is not None:
            return cls

    # Fallback to generic classes
    return AutoModel


def load_model(checkpoint: str, explicit_class: str | None):
    """Load a model given a checkpoint path/identifier."""
    config = AutoConfig.from_pretrained(checkpoint)
    model_cls = infer_model_class(config, explicit_class)

    load_attempts: Iterable[type] = (model_cls,)
    if model_cls is AutoModel:
        load_attempts = (
            AutoModel,
            AutoModelForSequenceClassification,
            AutoModelForQuestionAnswering,
        )

    last_error: Exception | None = None
    for candidate in load_attempts:
        try:
            model = candidate.from_pretrained(checkpoint, config=config)
            return model, config
        except Exception as exc:  # pragma: no cover - defensive fallback
            last_error = exc
            continue

    raise RuntimeError(f"Unable to load model from '{checkpoint}': {last_error}")


def strip_base_prefix(name: str, base_prefix: str | None) -> str:
    if base_prefix and name.startswith(base_prefix + "."):
        return name[len(base_prefix) + 1 :]
    return name


def classify_encoder_component(parts: List[str]) -> Tuple[int | None, str]:
    layer_idx: int | None = None
    cursor = 0

    if cursor < len(parts) and parts[cursor] in {"encoder", "transformer", "layers", "blocks"}:
        cursor += 1

    if cursor < len(parts) and parts[cursor] in {"layer", "layers", "block", "blocks"}:
        cursor += 1

    if cursor < len(parts) and parts[cursor].isdigit():
        layer_idx = int(parts[cursor])
        cursor += 1

    component_path = parts[cursor:]
    if not component_path:
        return layer_idx, "other"

    token = component_path[0].lower()
    if token == "attention":
        if len(component_path) >= 2:
            subtoken = component_path[1].lower()
            if subtoken == "self":
                return layer_idx, "attention_self"
            if subtoken == "output":
                if len(component_path) >= 3 and component_path[2].lower().startswith("layernorm"):
                    return layer_idx, "layer_norm"
                return layer_idx, "attention_output"
        return layer_idx, "attention"
    if token == "intermediate":
        return layer_idx, "ffn_intermediate"
    if token == "output":
        if len(component_path) >= 2 and component_path[1].lower().startswith("layernorm"):
            return layer_idx, "layer_norm"
        return layer_idx, "ffn_output"
    if token.startswith("layernorm"):
        return layer_idx, "layer_norm"
    return layer_idx, "other"


def bucketize_parameter(name: str, base_prefix: str | None) -> Tuple[str, int | None, str | None]:
    """Map a parameter name to a high-level component, optional layer index, and encoder sub-component."""
    suffix = strip_base_prefix(name, base_prefix)
    if not suffix:
        return "other", None, None

    parts = suffix.split(".")
    token = parts[0].lower()

    if base_prefix and token == base_prefix.lower():
        remainder = ".".join(parts[1:])
        if remainder:
            return bucketize_parameter(remainder, None)
        return "other", None, None

    # Default assignments
    bucket = token
    layer_idx: int | None = None
    encoder_component: str | None = None

    if token in {"embeddings", "shared", "tok_embeddings"}:
        bucket = "embeddings"
    elif token in {"encoder", "transformer", "layers", "blocks"}:
        bucket = "encoder"
        layer_idx, encoder_component = classify_encoder_component(parts)
    elif token in {"pooler", "pooling"}:
        bucket = "pooler"
    elif token in {"classifier", "pre_classifier", "score"}:
        bucket = "classifier_head"
    elif token in {"qa_outputs", "qa_classifier", "qa"}:
        bucket = "qa_head"
    elif token in {"lm_head", "cls", "language_model_head", "generator_lm_head"}:
        bucket = "lm_head"
    elif token in {"vocab_transform", "vocab_layer_norm"}:
        bucket = "lm_head"
    elif token in {"embeddings_project", "embeddings_projector"}:
        bucket = "embeddings"
    elif token == "dropout":
        bucket = "other"

    return bucket, layer_idx, encoder_component


def humanize_bucket(bucket: str) -> str:
    mapping: Dict[str, str] = {
        "embeddings": "Embeddings",
        "encoder": "Encoder",
        "pooler": "Pooler",
        "classifier_head": "Classifier head",
        "qa_head": "QA head",
        "lm_head": "LM head",
        "other": "Other",
    }
    return mapping.get(bucket, bucket.replace("_", " ").title())


def format_percentage(value: float, precision: int) -> str:
    return f"{value:.{precision}f}"


def collect_statistics(
    model,
    *,
    trainable_only: bool,
) -> Tuple[
    int,
    Dict[str, int],
    Dict[str, Dict[int, int]],
    Dict[str, int],
    Dict[int, Dict[str, int]],
]:
    base_prefix = getattr(model, "base_model_prefix", None)

    total_params = 0
    bucket_totals: Dict[str, int] = defaultdict(int)
    bucket_layers: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    encoder_components_total: Dict[str, int] = defaultdict(int)
    encoder_components_by_layer: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue
        numel = param.numel()
        total_params += numel

        bucket, layer_idx, encoder_component = bucketize_parameter(name, base_prefix)
        bucket_totals[bucket] += numel
        if layer_idx is not None:
            bucket_layers[bucket][layer_idx] += numel
        if bucket == "encoder" and encoder_component is not None:
            encoder_components_total[encoder_component] += numel
            if layer_idx is not None:
                encoder_components_by_layer[layer_idx][encoder_component] += numel

    return (
        total_params,
        bucket_totals,
        bucket_layers,
        encoder_components_total,
        encoder_components_by_layer,
    )


def print_report(
    *,
    total: int,
    bucket_totals: Dict[str, int],
    bucket_layers: Dict[str, Dict[int, int]],
    encoder_components_total: Dict[str, int],
    encoder_components_by_layer: Dict[int, Dict[str, int]],
    show_submodules: bool,
    precision: int,
) -> None:
    if total == 0:
        print("No parameters matched the selection criteria.")
        return

    print(f"Total parameters inspected: {total:,}")
    print("\nBreakdown by component:")

    sorted_buckets = sorted(bucket_totals.items(), key=lambda item: item[1], reverse=True)
    for bucket, count in sorted_buckets:
        share = 100.0 * count / total
        print(
            f"- {humanize_bucket(bucket):<20} {count:>12,} params "
            f"({format_percentage(share, precision)}%)"
        )

        if show_submodules and bucket in bucket_layers:
            layer_stats = bucket_layers[bucket]
            for layer_idx, layer_count in sorted(layer_stats.items()):
                layer_share = 100.0 * layer_count / total
                bucket_share = 100.0 * layer_count / count if count else 0.0
                print(
                    f"    • Layer {layer_idx:02d}: {layer_count:>12,} params "
                    f"({format_percentage(bucket_share, precision)}% of {humanize_bucket(bucket).lower()}, "
                    f"{format_percentage(layer_share, precision)}% overall)"
                )
                if bucket == "encoder" and layer_idx in encoder_components_by_layer:
                    component_stats = encoder_components_by_layer[layer_idx]
                    for component, component_count in sorted(
                        component_stats.items(), key=lambda item: item[1], reverse=True
                    ):
                        component_share_layer = 100.0 * component_count / layer_count if layer_count else 0.0
                        overall_share = 100.0 * component_count / total
                        print(
                            f"        · {component.replace('_', ' ').title():<18} {component_count:>12,} params "
                            f"({format_percentage(component_share_layer, precision)}% of layer, "
                            f"{format_percentage(overall_share, precision)}% overall)"
                        )

    if encoder_components_total:
        print("\nEncoder breakdown by component:")
        for component, count in sorted(
            encoder_components_total.items(), key=lambda item: item[1], reverse=True
        ):
            share = 100.0 * count / total if total else 0.0
            print(
                f"- {component.replace('_', ' ').title():<20} {count:>12,} params "
                f"({format_percentage(share, precision)}%)"
            )


def main() -> None:
    args = parse_args()
    model, config = load_model(args.checkpoint, args.class_name)
    model.eval()

    print(f"Model class: {model.__class__.__name__}")
    print(f"Model type: {getattr(config, 'model_type', 'unknown')}")
    architectures = getattr(config, "architectures", None)
    if architectures:
        print("Architectures declared:", ", ".join(architectures))
    base_prefix = getattr(model, "base_model_prefix", None)
    print(f"Base model prefix: {base_prefix if base_prefix else 'n/a'}")

    (
        total,
        bucket_totals,
        bucket_layers,
        encoder_components_total,
        encoder_components_by_layer,
    ) = collect_statistics(
        model,
        trainable_only=args.trainable_only,
    )
    print_report(
        total=total,
        bucket_totals=bucket_totals,
        bucket_layers=bucket_layers,
        encoder_components_total=encoder_components_total,
        encoder_components_by_layer=encoder_components_by_layer,
        show_submodules=args.show_submodules,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()
