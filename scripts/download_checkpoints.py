#!/usr/bin/env python3
"""Download and cache fine-tuned checkpoints for BERT and DistilBERT.

This helper script mirrors the structure expected by `main.py` by materialising
Hugging Face model repositories inside the local `checkpoints/` directory. Each
checkpoint folder will contain the `config.json`, `pytorch_model.bin`, and
associated tokenizer files, making it straightforward to run the pruning
pipeline offline.

"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Iterable, Literal

from transformers import (
    AutoModel,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

default_tasks: Dict[str, Literal["sequence_classification", "question_answering"]] = {
    "mnli": "sequence_classification",
    "qqp": "sequence_classification",
    "qnli": "sequence_classification",
    "sst2": "sequence_classification",
    "stsb": "sequence_classification",
    "mrpc": "sequence_classification",
    "squad": "question_answering",
    "squad_v2": "question_answering",
}

# Publicly available fine-tuned checkpoints hosted on the Hugging Face Hub.
model_task_to_repo: Dict[str, Dict[str, str]] = {
    "bert-base-uncased": {
        "mnli": "textattack/bert-base-uncased-MNLI",
        "qqp": "textattack/bert-base-uncased-QQP",
        "qnli": "textattack/bert-base-uncased-QNLI",
        "sst2": "textattack/bert-base-uncased-SST-2",
        "stsb": "textattack/bert-base-uncased-STS-B",
        "mrpc": "textattack/bert-base-uncased-MRPC",
        "squad": "csarron/bert-base-uncased-squad-v1",
        "squad_v2": "deepset/bert-base-uncased-squad2",
    },
    "distilbert-base-uncased": {
        "mnli": "textattack/distilbert-base-uncased-MNLI",
        "qqp": "textattack/distilbert-base-uncased-QQP",
        "qnli": "textattack/distilbert-base-uncased-QNLI",
        "sst2": "textattack/distilbert-base-uncased-SST-2",
        "stsb": "tomaarsen/distilbert-base-uncased-sts",
        "mrpc": "textattack/distilbert-base-uncased-MRPC",
        "squad": "distilbert-base-uncased-distilled-squad",
        "squad_v2": "twmkn9/distilbert-base-uncased-squad2",
    },
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(model_task_to_repo.keys()),
        help="Base model identifiers to download (defaults to both BERT-base and DistilBERT)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(default_tasks.keys()),
        choices=list(default_tasks.keys()),
        help="Tasks to download. Defaults to every supported GLUE/SQuAD task.",
    )
    parser.add_argument(
        "--output-dir",
        default="/n/netscratch/sham_lab/Everyone/tdatta/pruning/checkpoints/",
        type=Path,
        help="Root directory where checkpoints will be stored.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory (TRANSFORMERS_CACHE).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face access token for private repositories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already-downloaded checkpoints.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help=(
            "Skip download when no repository mapping is provided for a given "
            "model/task combination (instead of falling back to the base model)."
        ),
    )
    return parser.parse_args()


def resolve_repo_id(model_name: str, task: str) -> str | None:
    task_to_repo = model_task_to_repo.get(model_name, {})
    return task_to_repo.get(task)


def ensure_clean_dir(target_dir: Path, overwrite: bool) -> None:
    if target_dir.exists() and overwrite:
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)


def save_metadata(target_dir: Path, payload: Dict[str, str]) -> None:
    metadata_path = target_dir / "metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def download_checkpoint(
    model_name: str,
    task: str,
    repo_id: str,
    target_dir: Path,
    task_type: Literal["sequence_classification", "question_answering"],
    *,
    cache_dir: Path | None = None,
    hf_token: str | None = None,
) -> None:
    hf_kwargs = {
        "cache_dir": cache_dir,
        "use_auth_token": hf_token,
    }
    hf_kwargs = {k: v for k, v in hf_kwargs.items() if v is not None}

    model_cls = (
        AutoModelForQuestionAnswering
        if task_type == "question_answering"
        else AutoModelForSequenceClassification
    )

    logging.info("Downloading %s (%s) from %s", model_name, task, repo_id)
    model = model_cls.from_pretrained(repo_id, **hf_kwargs)
    ensure_clean_dir(target_dir, overwrite=True)
    model.save_pretrained(target_dir)

    # Persist the tokenizer locally so that the pipeline can run fully offline.
    tokenizer_sources: Iterable[str] = (repo_id, model_name)
    tokenizer = None
    for source in tokenizer_sources:
        try:
            tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, **hf_kwargs)
            break
        except Exception as exc:  # pragma: no cover - emergency fallback logging
            logging.warning("Failed to fetch tokenizer from %s (%s)", source, exc)
            continue

    if tokenizer is None:
        raise RuntimeError(
            f"Unable to load a tokenizer for {model_name}. Please specify an alternative repository."
        )

    tokenizer.save_pretrained(target_dir)
    save_metadata(
        target_dir,
        {
            "base_model": model_name,
            "task": task,
            "repository": repo_id,
            "task_type": task_type,
        },
    )


def download_base_models(
    model_name: str,
    target_root: Path,
    *,
    cache_dir: Path | None = None,
    hf_token: str | None = None,
) -> None:
    hf_kwargs = {
        "cache_dir": cache_dir,
        "use_auth_token": hf_token,
    }
    hf_kwargs = {k: v for k, v in hf_kwargs.items() if v is not None}

    base_spec = {
        "sequence_classification": AutoModelForSequenceClassification,
        "question_answering": AutoModelForQuestionAnswering,
    }

    for task_type, model_cls in base_spec.items():
        target_dir = target_root / f"base_{task_type}"
        ensure_clean_dir(target_dir, overwrite=True)

        logging.info("Downloading base %s weights for %s", task_type, model_name)
        model = model_cls.from_pretrained(model_name, **hf_kwargs)
        model.save_pretrained(target_dir)

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **hf_kwargs)
        tokenizer.save_pretrained(target_dir)

        save_metadata(
            target_dir,
            {
                "base_model": model_name,
                "task": f"base_{task_type}",
                "repository": model_name,
                "task_type": task_type,
            },
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    cache_dir = args.cache_dir.resolve() if args.cache_dir is not None else None
    output_dir: Path = args.output_dir.resolve()

    for model_name in args.models:
        if model_name not in model_task_to_repo:
            logging.warning(
                "Model %s has no predefined repository mapping; downloads will fall back to the base identifier.",
                model_name,
            )

        base_root_dir = output_dir / model_name
        download_base_models(
            model_name,
            base_root_dir,
            cache_dir=cache_dir,
            hf_token=args.hf_token,
        )

        for task in args.tasks:
            task_type = default_tasks[task]
            repo_id = resolve_repo_id(model_name, task)

            if repo_id is None:
                if args.skip_missing:
                    logging.info(
                        "Skipping %s (%s) because no repository mapping is defined and --skip-missing was used.",
                        model_name,
                        task,
                    )
                    continue
                logging.warning(
                    "No repository mapping for %s (%s); falling back to the base model identifier.",
                    model_name,
                    task,
                )
                repo_id = model_name

            target_dir = output_dir / model_name / task
            if target_dir.exists() and not args.overwrite:
                logging.info("Skipping existing checkpoint at %s", target_dir)
                continue

            ensure_clean_dir(target_dir, overwrite=args.overwrite)
            download_checkpoint(
                model_name,
                task,
                repo_id,
                target_dir,
                task_type,
                cache_dir=cache_dir,
                hf_token=args.hf_token,
            )

    logging.info("All requested checkpoints are ready under %s", output_dir)


if __name__ == "__main__":
    main()
