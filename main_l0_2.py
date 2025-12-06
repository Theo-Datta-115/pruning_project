"""
Training script using L0Module for structured attention head + FFN neuron pruning.

This file follows the logging and input structure of main.py while implementing
L0-based pruning similar to the CoFiTrainer approach.

Base command:
python main_l0_2.py --model_name bert-base-uncased --task_name qqp --use_base_model --prune --target_sparsity 0.5
"""

import argparse
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)

from dataset.glue import glue_dataset, max_seq_length, avg_seq_length
from dataset.squad import squad_dataset
from evaluate_model.nlp import test_accuracy
from utils.l0 import L0Module
from utils.model_loader import load_model_and_tokenizer
from utils.arch import get_layers


class L0ModuleNonHierarchical(L0Module):
    """L0Module wrapper that computes parameter counts without layer-level masks."""
    
    def get_num_parameters_and_constraint(self):
        """Override to handle missing layer-level masks (head_layer, mlp)."""
        num_parameters = 0
        
        # For heads: just use per-head masks without layer-level multiplication
        if "head" in self.types:
            head_score = 1 - self.cdf_qz(0, self.head_loga)  # [num_layers, num_heads]
            num_parameters += torch.sum(head_score) * self.parameters_per_dim["head"]
        
        # For FFN: just use per-intermediate masks without layer-level multiplication
        if "intermediate" in self.types:
            int_score = 1 - self.cdf_qz(0, self.int_loga)  # [num_layers, intermediate_size]
            num_parameters += torch.sum(int_score) * self.parameters_per_dim["intermediate"]
        
        return num_parameters
    
    def lagrangian_regularization(self, pruned_steps):
        """Override to use the non-hierarchical parameter counting."""
        target_sparsity = self.target_sparsity
        expected_size = self.get_num_parameters_and_constraint()
        expected_sparsity = 1 - expected_size / self.prunable_model_size
        
        if self.lagrangian_warmup > 0:
            target_sparsity = self.get_target_sparsity(pruned_steps)
        
        lagrangian_loss = (
            self.lambda_1 * (expected_sparsity - target_sparsity)
            + self.lambda_2 * (expected_sparsity - target_sparsity) ** 2
        )
        return lagrangian_loss, expected_sparsity, target_sparsity


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--task_name", type=str, required=True, choices=[
    "mnli", "qqp", "qnli", "sst2", "stsb", "mrpc", "squad", "squad_v2",
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
parser.add_argument("--num_epochs", type=int, default=12)
parser.add_argument('--log_loss_every', type=int, default=5)
parser.add_argument('--prune', action='store_true', help='Enable pruning with L0Module')
parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
parser.add_argument('--name', type=str, default=None, help='W&B run name (defaults to auto-generated)')

# L0-specific arguments
parser.add_argument('--target_sparsity', type=float, default=0.5, help='Target sparsity level')
parser.add_argument('--start_sparsity', type=float, default=0.0, help='Starting sparsity level')
parser.add_argument('--prepruning_finetune_steps', type=int, default=100, 
                    help='Steps to train before starting L0 pruning')
parser.add_argument('--lagrangian_warmup_epochs', type=int, default=2, 
                    help='Epochs to warmup lagrangian after prepruning')
parser.add_argument('--l0_temperature', type=float, default=2./3., 
                    help='Temperature for L0 concrete distribution')
parser.add_argument('--reg_learning_rate', type=float, default=0.01, 
                    help='Learning rate for L0 masks and lambdas')


def apply_l0_masks_to_model(model, l0_module, zs):
    """
    Apply L0 masks to the model's attention and FFN layers.
    This function fills the model with the sampled masks from L0Module.
    """
    # Extract masks from zs dictionary
    head_z = zs.get('head_z', None)  # [num_layers, 1, num_heads, 1, 1]
    intermediate_z = zs.get('intermediate_z', None)  # [num_layers, 1, 1, intermediate_size]
    
    layers = get_layers(model)
    
    for layer_idx, layer in enumerate(layers):
        # Apply head masks to attention
        if head_z is not None:
            # Store mask on the attention module for use in forward pass
            mask = head_z[layer_idx].squeeze()  # [num_heads]
            layer.attention.self.head_mask = mask.view(1, -1, 1, 1)
        
        # Apply FFN masks to intermediate activations
        if intermediate_z is not None:
            mask = intermediate_z[layer_idx].squeeze()  # [intermediate_size]
            # Store on the output dense layer to mask intermediate activations
            layer.output.intermediate_mask = mask.view(1, 1, -1)


def wrap_model_with_l0_masks(model):
    """
    Wrap model forward passes to use stored L0 masks.
    This modifies the model's forward methods to apply masks during training.
    """
    layers = get_layers(model)
    
    for layer in layers:
        # Wrap attention forward
        if not hasattr(layer.attention.self, '_l0_wrapped'):
            original_attn_forward = layer.attention.self.forward
            
            def make_attn_forward(orig_forward):
                def forward(self, hidden_states, attention_mask=None, head_mask=None, 
                           encoder_hidden_states=None, encoder_attention_mask=None,
                           past_key_value=None, output_attentions=False):
                    # Use stored mask if available
                    if hasattr(self, 'head_mask') and self.training:
                        effective_mask = self.head_mask
                    else:
                        effective_mask = head_mask
                    
                    return orig_forward(
                        hidden_states, attention_mask, effective_mask,
                        encoder_hidden_states, encoder_attention_mask,
                        past_key_value, output_attentions
                    )
                return forward
            
            layer.attention.self.forward = make_attn_forward(original_attn_forward).__get__(
                layer.attention.self, layer.attention.self.__class__
            )
            layer.attention.self._l0_wrapped = True
        
        # Wrap FFN output forward
        if not hasattr(layer.output, '_l0_wrapped'):
            original_output_forward = layer.output.forward
            
            def make_output_forward(orig_forward):
                def forward(self, hidden_states, input_tensor):
                    # Apply mask to intermediate activations if available
                    if hasattr(self, 'intermediate_mask') and self.training:
                        hidden_states = hidden_states * self.intermediate_mask
                    return orig_forward(hidden_states, input_tensor)
                return forward
            
            layer.output.forward = make_output_forward(original_output_forward).__get__(
                layer.output, layer.output.__class__
            )
            layer.output._l0_wrapped = True


def main():
    args = parser.parse_args()
    IS_SQUAD = "squad" in args.task_name
    IS_LARGE = "large" in args.model_name
    seq_len = 170 if IS_SQUAD else avg_seq_length(args.task_name)

    # Initialize wandb if enabled
    if args.wandb:
        os.environ["WANDB_DIR"] = "/n/home03/tdatta/pruning_project/scripts/wandb_logs"
        import wandb
        wandb.init(
            entity="harvardml",
            project="td-pruning",
            name=f"{args.model_name}_{args.task_name}_{args.name}" if args.name else None,
            config=vars(args),
        )
        logger.info(f"Weights & Biases logging enabled (project: td-pruning-l0)")
    else:
        wandb = None

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

    # Sample the examples to be used for search (following main.py structure)
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

    if args.prune:
        logger.info("Pruning enabled - using L0Module for structured pruning")
        logger.info("Pruning at attention head + FFN neuron granularity (no layer-level masks)")
        
        # Initialize custom L0Module WITHOUT layer-level masks
        # Only prunes at attention head and FFN neuron granularity
        l0_module = L0ModuleNonHierarchical(
            config=config,
            droprate_init=0.5,
            temperature=args.l0_temperature,
            lagrangian_warmup=0,  # Will be set later based on epochs
            start_sparsity=args.start_sparsity,
            target_sparsity=args.target_sparsity,
            pruning_type="structured_heads+structured_mlp",  # Only per-unit masks, no layer-level
        ).cuda()
        
        # Wrap model to use L0 masks
        wrap_model_with_l0_masks(model)
        
        # Collect parameters
        l0_params = list(l0_module.parameters())
        model_params = [p for p in model.parameters() if p.requires_grad]
        
        logger.info(f"L0Module parameters: {sum(p.numel() for p in l0_params)}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model_params)}")
    else:
        logger.info("Pruning disabled - training without masks")
        l0_module = None
        l0_params = []
        model_params = [p for p in model.parameters() if p.requires_grad]

    # Use smaller batch size for SST2 and QNLI to prevent overfitting
    if args.task_name in ['sst2', 'qnli']:
        train_batch_size = 64
    else:
        train_batch_size = 256
    
    train_dataloader = DataLoader(
        training_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Calculate training steps and set lagrangian warmup
    num_update_steps_per_epoch = len(train_dataloader)
    total_steps = args.num_epochs * num_update_steps_per_epoch
    
    if args.prune:
        lagrangian_warmup_steps = args.lagrangian_warmup_epochs * num_update_steps_per_epoch
        l0_module.set_lagrangian_warmup_steps(lagrangian_warmup_steps)
        logger.info(f"Prepruning finetune steps: {args.prepruning_finetune_steps}")
        logger.info(f"Lagrangian warmup steps: {lagrangian_warmup_steps}")

    # Optimizers - use THREE separate optimizers like CoFiTrainer
    # This is critical for proper lambda behavior with negative LR
    optimizer = torch.optim.AdamW([
        {"params": model_params, "lr": 5e-5, "weight_decay": 0.01},
    ])
    
    if args.prune:
        # Separate L0 mask parameters from lambda parameters
        lambda_params = [l0_module.lambda_1, l0_module.lambda_2]
        lambda_param_ids = {id(p) for p in lambda_params}
        mask_params = [p for p in l0_params if id(p) not in lambda_param_ids]
        
        # L0 optimizer for mask parameters
        l0_optimizer = torch.optim.AdamW(
            mask_params, lr=args.reg_learning_rate, weight_decay=0.0
        )
        
        # Lagrangian optimizer for gradient ascent
        # We'll negate gradients manually after backward pass
        lagrangian_optimizer = torch.optim.AdamW(
            lambda_params, lr=args.reg_learning_rate, weight_decay=0.0
        )
    else:
        l0_optimizer = None
        lagrangian_optimizer = None
    
    model.train()
    global_step = 0
    start_prune = False  # Flag to track when to start pruning

    # Learning rate scheduler (only for main model optimizer)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(training_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_steps}")

    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        model.train()
        if args.prune:
            l0_module.train()
        
        epoch_loss = 0.0
        for idx, batch in enumerate(train_dataloader):
            for key, value in batch.items():
                batch[key] = value.to("cuda", non_blocking=True)

            # Check if we should start pruning (following CoFiTrainer pattern)
            if args.prune and not start_prune and global_step >= args.prepruning_finetune_steps:
                start_prune = True
                logger.info(f"Starting L0 regularization at step {global_step}")

            # Sample masks from L0Module (only if pruning has started)
            if args.prune and start_prune:
                zs = l0_module.forward(training=True)
                apply_l0_masks_to_model(model, l0_module, zs)
            
            outputs = model(**batch)
            
            if args.prune and start_prune:
                # Compute Lagrangian loss (use adjusted step count)
                lagrangian_loss, expected_sparsity, target_sparsity_current = \
                    l0_module.lagrangian_regularization(global_step - args.prepruning_finetune_steps)
            else:
                lagrangian_loss = torch.tensor(0.0)
                expected_sparsity = torch.tensor(0.0)
                target_sparsity_current = 0.0

            loss = outputs.loss + lagrangian_loss
            loss.backward()
            
            # Negate lambda gradients immediately after backward for gradient ascent
            if args.prune and start_prune:
                for param in [l0_module.lambda_1, l0_module.lambda_2]:
                    if param.grad is not None:
                        param.grad.neg_()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if args.prune:
                torch.nn.utils.clip_grad_norm_(l0_module.parameters(), max_norm=1.0)
            
            optimizer.step()
            if args.prune and start_prune:
                l0_optimizer.step()
                lagrangian_optimizer.step()
            
            scheduler.step()
            
            # Constrain L0 parameters to reasonable range
            if args.prune:
                l0_module.constrain_parameters()
            
            optimizer.zero_grad(set_to_none=True)
            if args.prune and start_prune:
                l0_optimizer.zero_grad(set_to_none=True)
                lagrangian_optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            global_step += 1

            if args.log_loss_every > 0 and global_step % args.log_loss_every == 0:
                if args.prune:
                    logger.info(
                        f"Step {global_step}: loss {outputs.loss.item():.4f}, "
                        f"lagrangian {lagrangian_loss.item():.4f}, "
                        f"sparsity {expected_sparsity.item():.4f} (target {target_sparsity_current:.4f}), "
                        f"λ1={l0_module.lambda_1.item():.4f}, λ2={l0_module.lambda_2.item():.4f}"
                    )
                    
                    if args.wandb:
                        wandb.log({
                            "train/loss": outputs.loss.item(),
                            "train/lagrangian_loss": lagrangian_loss.item(),
                            "train/total_loss": loss.item(),
                            "train/expected_sparsity": expected_sparsity.item(),
                            "train/target_sparsity": target_sparsity_current,
                            "train/lambda_1": l0_module.lambda_1.item(),
                            "train/lambda_2": l0_module.lambda_2.item(),
                            "global_step": global_step,
                            "epoch": epoch,
                        })
                else:
                    logger.info(f"Step {global_step}: training loss {outputs.loss.item():.4f}")
                    
                    if args.wandb:
                        wandb.log({
                            "train/loss": outputs.loss.item(),
                            "global_step": global_step,
                            "epoch": epoch,
                        })

        epoch_end = time.time()
        logger.info(f"Epoch {epoch + 1} finished. Took {round(epoch_end - epoch_start, 2)} seconds.")

        # Evaluate at the end of each epoch
        model.eval()
        if args.prune:
            l0_module.eval()
        
        if args.prune:
            # Get deterministic masks for evaluation
            zs = l0_module.forward(training=False)
            head_z = zs.get('head_z', None)
            intermediate_z = zs.get('intermediate_z', None)
            
            # Convert to binary masks for evaluation
            # The deterministic masks are already binary (0 or soft values)
            # Threshold at 0.5 to convert soft masks to hard binary
            if head_z is not None:
                head_mask = (head_z.squeeze() > 0.5).float()  # [num_layers, num_heads]
            else:
                head_mask = None
            
            if intermediate_z is not None:
                ffn_mask = (intermediate_z.squeeze() > 0.5).float()  # [num_layers, intermediate_size]
            else:
                ffn_mask = None
            
            # Log actual sparsity
            if head_mask is not None:
                actual_head_sparsity = 1 - head_mask.mean().item()
                logger.info(f"Actual head sparsity: {actual_head_sparsity:.4f}")
            if ffn_mask is not None:
                actual_ffn_sparsity = 1 - ffn_mask.mean().item()
                logger.info(f"Actual FFN sparsity: {actual_ffn_sparsity:.4f}")
        else:
            head_mask = None
            ffn_mask = None
        
        epoch_test_acc = test_accuracy(model, head_mask, ffn_mask, tokenizer, args.task_name)
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs} - Test accuracy: {epoch_test_acc:.4f}")
        
        if args.wandb:
            log_dict = {
                "eval/accuracy": epoch_test_acc,
                "epoch": epoch,
                "global_step": global_step,
            }
            if args.prune:
                if head_mask is not None:
                    log_dict["eval/actual_head_sparsity"] = actual_head_sparsity
                if ffn_mask is not None:
                    log_dict["eval/actual_ffn_sparsity"] = actual_ffn_sparsity
            wandb.log(log_dict)
    
    if args.wandb:
        wandb.finish()

    # Save the masks (only if pruning is enabled)
    if args.prune:
        # Create directory based on wandb run name if available, otherwise use output_dir
        if args.wandb and args.name:
            save_dir = os.path.join(args.output_dir, args.name)
        elif args.wandb:
            save_dir = os.path.join(args.output_dir, wandb.run.name)
        else:
            save_dir = args.output_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        if head_mask is not None:
            torch.save(head_mask.cpu(), os.path.join(save_dir, "head_mask.pt"))
        if ffn_mask is not None:
            torch.save(ffn_mask.cpu(), os.path.join(save_dir, "ffn_mask.pt"))
        
        torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pt"))
        torch.save(l0_module.state_dict(), os.path.join(save_dir, "l0_module.pt"))
        
        torch.save({
            'lambda_1': l0_module.lambda_1.item(),
            'lambda_2': l0_module.lambda_2.item(),
        }, os.path.join(save_dir, "lagrange_multipliers.pt"))

        logger.info(f"Masks saved to {save_dir}")
        if args.prune:
            logger.info(f"Final Lagrange multipliers: λ1={l0_module.lambda_1.item():.4f}, λ2={l0_module.lambda_2.item():.4f}")

if __name__ == "__main__":
    main()
