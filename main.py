"""
My short term goals here: 
- Re-run the paper results, validate them, validate their pipeline works
- Set up an actual train pipeline that retrains on BERT data, and validate that it doesn't significantly change the results 
    - Use the wiki data from BERT training, raw = load_dataset("wikipedia", "20220301.en", split="train")
- Try adding a new L2 training objective on the model as a whole, see what happens
- Add wandb logging, potentially
- Test what it looks like to finetune on a pre-finetuned model versus a new model (i.e. does compressing while finetuning help, or should you compress post-tuning)
- consider adding wandb for training
- there is a hook (non-gradient, short-term) masking emthod implemented in utils and evaluate. This is fine for my purposes, as long as I actually train the model with new NN parameters, and then extract them and pass them to the original model architecture at test time. To do this, I should implement the ability to use hooks on both ouput and input ffn layers.

Base command: 

python main.py --model_name bert-base-uncased  --task_name qqp --use_base_model
python baseline.py --model_name bert-base-uncased  --task_name qqp --constraint 0.5
python scripts/analyze_param_usage.py /n/netscratch/sham_lab/Everyone/tdatta/pruning/checkpoints/bert-base-uncased/qqp
python scripts/download_checkpoints.py

"""

import argparse
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)

from dataset.glue import glue_dataset, max_seq_length, avg_seq_length
from dataset.squad import squad_dataset
from efficiency.mac import compute_mask_mac
from efficiency.latency import estimate_latency
from prune.fisher import collect_mask_grads
from prune.search import search_mac, search_latency
from prune.rearrange import rearrange_mask
from prune.rescale import rescale_mask
from evaluate.nlp import test_accuracy
from utils.schedule import get_pruning_schedule
from learned_prune.trainable_masks import make_masks_trainable
from utils.model_loader import load_model_and_tokenizer


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--task_name", type=str, required=True, choices=[
    "mnli",
    "qqp",
    "qnli",
    "sst2",
    "stsb",
    "mrpc",
    "squad",
    "squad_v2",
])
parser.add_argument(
    "--ckpt_dir",
    type=str,
    default=None,
    help=(
        "Path to the fine-tuned checkpoint directory. Defaults to "
        "`checkpoints/<model_name>/<task_name>` if not provided."
    ),
)
parser.add_argument("--output_dir", type=str, default="/n/netscratch/sham_lab/Everyone/tdatta/pruning/outputs/")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--num_samples", type=int, default=2048)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--use_base_model",
    action="store_true",
    help="Load the base pretrained weights for `--model_name` instead of a fine-tuned checkpoint.",
)
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument('--log_loss_every', type=int, default=5)


def main():
    args = parser.parse_args()
    IS_SQUAD = "squad" in args.task_name
    IS_LARGE = "large" in args.model_name
    seq_len = 170 if IS_SQUAD else avg_seq_length(args.task_name)

    default_root = "/n/netscratch/sham_lab/Everyone/tdatta/pruning/checkpoints/"
    config, model, tokenizer, model_source = load_model_and_tokenizer(
        model_name=args.model_name,
        task_name=args.task_name,
        ckpt_dir=args.ckpt_dir,
        use_base_model=args.use_base_model,
        default_root=default_root,
    )

    # Create the output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "outputs",
            args.model_name,
            args.task_name,
            f"seed_{args.seed}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Initiate the logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
        ],
    )
    logger.info(args)

    # Set a GPU and the experiment seed
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)
    logger.info(f"Seed number: {args.seed}")

    # Load the finetuned model and the corresponding tokenizer
    # Load the training dataset
    if IS_SQUAD:
        training_dataset = squad_dataset(
            args.task_name,
            tokenizer,
            training=True,
            max_seq_len=384,
            pad_to_max=False,
        )
    else:
        training_dataset = glue_dataset(
            args.task_name,
            tokenizer,
            training=True,
            max_seq_len=max_seq_length(args.task_name),
            pad_to_max=False,
        )

    # Sample the examples to be used for search
    collate_fn = DataCollatorWithPadding(tokenizer)
    sample_dataset = Subset(
        training_dataset,
        np.random.choice(len(training_dataset), args.num_samples).tolist(),
    )
    sample_batch_size = int((12 if IS_SQUAD else 32) * (0.5 if IS_LARGE else 1))
    sample_dataloader = DataLoader(
        sample_dataset,
        batch_size=sample_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )

    # Prepare the model
    model = model.cuda()

    full_head_mask = torch.randn(config.num_hidden_layers, config.num_attention_heads).cuda()
    full_ffn_intermediate_mask = torch.randn(config.num_hidden_layers, config.hidden_size).cuda()
    full_ffn_output_mask = torch.randn(config.num_hidden_layers, config.intermediate_size).cuda()


    masked_model = make_masks_trainable(
        model,
        head_mask=full_head_mask,
        ffn_intermediate_mask=full_ffn_intermediate_mask,
        ffn_output_mask=full_ffn_output_mask,
    )

    mask_params = [
        param
        for param in (
            masked_model.head_mask,
            masked_model.ffn_intermediate_mask,
            masked_model.ffn_output_mask,
        )
        if param is not None
    ]
    mask_param_ids = {id(param) for param in mask_params}
    base_params = [
        param for param in model.parameters() if id(param) not in mask_param_ids
    ]

    masked_train_dataloader = DataLoader(
        training_dataset,
        batch_size=256,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Moodel finetuning
    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": 2e-5, "weight_decay": 0.01},
            {"params": mask_params, "lr": 1e-2, "weight_decay": 0.0},
        ]
    )
    model.train()
    global_step = 0

    m = model.trainable_ffn_intermediate_mask
    print("requires_grad:", m.requires_grad)
    ids = {id(p) for g in optimizer.param_groups for p in g['params']}
    print("in optimizer:", id(m) in ids)

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        for batch in masked_train_dataloader:
            for key, value in batch.items():
                batch[key] = value.to("cuda", non_blocking=True)

            outputs = model(**batch)
            
            masks = torch.sigmoid(model.trainable_ffn_intermediate_mask)
            sparsity_loss = torch.mean(masks)# + torch.mean(model.trainable_ffn_output_mask) + torch.mean(model.trainable_head_mask)
            
            
            quantization_loss = torch.mean(-torch.log(masks) * masks - torch.log(1 - masks) * (1 - masks))
            # quantization_loss = (4 * masks * (1 - masks)).mean()

            loss = outputs.loss + sparsity_loss * 0.1 + quantization_loss * 0.01
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            global_step += 1

            if args.log_loss_every > 0 and global_step % args.log_loss_every == 0:
                logger.info(f"Step {global_step} of {len(masked_train_dataloader)}: training loss {outputs.loss.item():.4f}, sparsity loss {sparsity_loss.item():.4f}")
                print(torch.sigmoid(model.trainable_ffn_intermediate_mask))

    model.eval()

    start = time.time()

    head_mask = masked_model.head_mask.detach()
    ffn_intermediate_mask = masked_model.ffn_intermediate_mask.detach()
    ffn_output_mask = masked_model.ffn_output_mask.detach()

    # Print the pruning time
    end = time.time()

    logger.info(f"{args.task_name} Pruning time (s): {end - start}")

    # Evaluate the accuracy
    test_acc = test_accuracy(model, head_mask, ffn_intermediate_mask, ffn_output_mask, tokenizer, args.task_name)
    logger.info(f"{args.task_name} Test accuracy: {test_acc:.4f}")

    # Save the masks
    torch.save(head_mask.cpu(), os.path.join(args.output_dir, "head_mask.pt"))
    torch.save(ffn_intermediate_mask.cpu(), os.path.join(args.output_dir, "ffn_intermediate_mask.pt"))
    torch.save(ffn_output_mask.cpu(), os.path.join(args.output_dir, "ffn_output_mask.pt"))

if __name__ == "__main__":
    main()
