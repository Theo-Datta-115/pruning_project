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
    get_linear_schedule_with_warmup,
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
from utils.schedule import get_pruning_schedule, gumbel_sigmoid, get_gumbel_temperature
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
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument('--log_loss_every', type=int, default=5)
parser.add_argument('--prune', action='store_true', help='Enable pruning with trainable masks')
parser.add_argument('--wandb', action='store_true', help='W&B project name (enables W&B logging if provided)')
parser.add_argument('--name', type=str, default=None, help='W&B run name (defaults to auto-generated)')
parser.add_argument('--freeze', action='store_true', help='Freeze the model parameters')
parser.add_argument('--quantization_loss', type=float, default=0.01, help='Quantization loss weight')
parser.add_argument('--sparsity_loss', type=float, default=0.1, help='Sparsity loss weight')
parser.add_argument('--hard_init', action='store_true', help='Hard Initialization of the Masks')
parser.add_argument('--masks_LR', type=float, default=1e-2, help='Learning rate for masks')
parser.add_argument('--learn_masks', type=str, default='ffn_int', 
                    choices=['head', 'ffn_int', 'ffn_out', 'all'],
                    help='Which masks to learn: head, ffn_int, ffn_out, or all')
parser.add_argument('--gumbel_temp_start', type=float, default=5.0, help='Starting Gumbel temperature')
parser.add_argument('--gumbel_temp_end', type=float, default=1.0, help='Ending Gumbel temperature')
parser.add_argument('--gumbel_temp_anneal', type=str, default='linear', 
                    choices=['linear', 'exponential', 'constant'],
                    help='Temperature annealing schedule')
parser.add_argument('--layerwise_train', action='store_true', 
                    help='Train each BERT layer sequentially instead of all at once')
parser.add_argument('--layerwise_order', type=str, default='front_to_back',
                    choices=['front_to_back', 'back_to_front', 'random'],
                    help='Order for layer-wise training')

def compute_mask_losses(model, learn_head, learn_ffn_int, learn_ffn_out, temperature):
    """Compute sparsity and quantization losses for active masks."""
    sparsity_loss = 0.0
    quantization_loss = 0.0
    num_active = 0
    
    for mask_name, mask_tensor, is_active in [
        ('head', model.trainable_head_mask, learn_head),
        ('ffn_int', model.trainable_ffn_intermediate_mask, learn_ffn_int),
        ('ffn_out', model.trainable_ffn_output_mask, learn_ffn_out),
    ]:
        if is_active:
            mask_sigmoid = torch.sigmoid(mask_tensor)
            sparsity_loss += torch.mean(mask_sigmoid)
            
            mask_clamped = torch.clamp(mask_sigmoid, 1e-7, 1 - 1e-7)
            quantization_loss += torch.mean(
                -torch.log(mask_clamped) * mask_clamped - 
                torch.log(1 - mask_clamped) * (1 - mask_clamped)
            )
            num_active += 1
    
    if num_active > 0:
        sparsity_loss /= num_active
        quantization_loss /= num_active
    
    return sparsity_loss, quantization_loss

def training_step(batch, model, optimizer, args, global_step, total_steps, learn_head, learn_ffn_int, learn_ffn_out, layer_idx=None):
    """Execute a single training step. Returns (loss, outputs, sparsity_loss, quantization_loss, current_temp)."""
    for key, value in batch.items():
        batch[key] = value.to("cuda", non_blocking=True)

    # Update Gumbel temperature
    if args.prune:
        current_temp = get_gumbel_temperature(
            global_step, total_steps, args.gumbel_temp_start, 
            args.gumbel_temp_end, args.gumbel_temp_anneal
        )
        model.gumbel_temperature = current_temp
    else:
        current_temp = None
    
    outputs = model(**batch)
    
    if args.prune:
        sparsity_loss, quantization_loss = compute_mask_losses(
            model, learn_head, learn_ffn_int, learn_ffn_out, current_temp
        )
    else:
        sparsity_loss = torch.tensor(0.0)
        quantization_loss = torch.tensor(0.0)

    loss = outputs.loss + sparsity_loss * args.sparsity_loss + quantization_loss * args.quantization_loss
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    return loss, outputs, sparsity_loss, quantization_loss, current_temp

def log_training_step(logger, wandb, args, global_step, outputs, loss, sparsity_loss, quantization_loss, current_temp, model, learn_head, learn_ffn_int, learn_ffn_out, epoch=None, layer_idx=None):
    """Log training metrics."""
    if args.prune:
        log_msg = f"Step {global_step}: loss {outputs.loss.item():.4f}, sparsity {sparsity_loss.item():.4f}, quant {quantization_loss.item():.4f}, temp {current_temp:.4f}"
        if layer_idx is not None:
            log_msg = f"Layer {layer_idx}, " + log_msg
        logger.info(log_msg)
        
        if args.wandb:
            log_dict = {
                "train/loss": outputs.loss.item(),
                "train/sparsity_loss": sparsity_loss.item(),
                "train/quantization_loss": quantization_loss.item(),
                "train/total_loss": loss.item(),
                "train/gumbel_temperature": current_temp,
                "global_step": global_step,
            }
            if epoch is not None:
                log_dict["epoch"] = epoch
            if layer_idx is not None:
                log_dict["current_layer"] = layer_idx
            
            # Log mask statistics
            for mask_name, mask_tensor, is_active in [
                ('head', model.trainable_head_mask, learn_head),
                ('ffn_int', model.trainable_ffn_intermediate_mask, learn_ffn_int),
                ('ffn_out', model.trainable_ffn_output_mask, learn_ffn_out),
            ]:
                if is_active:
                    if layer_idx is not None:
                        # For layerwise, log specific layer
                        m_sig = torch.sigmoid(mask_tensor[layer_idx])
                        log_dict[f"masks/layer_{layer_idx}_{mask_name}_mean"] = m_sig.mean().item()
                        log_dict[f"masks/layer_{layer_idx}_{mask_name}_pct_below_0.5"] = 100 * (m_sig < 0.5).float().mean().item()
                    else:
                        # For standard, log all layers
                        m_sig = torch.sigmoid(mask_tensor)
                        log_dict[f"masks/{mask_name}_mean"] = m_sig.mean().item()
                        log_dict[f"masks/{mask_name}_pct_below_0.5"] = 100 * (m_sig < 0.5).float().mean().item()
                        log_dict[f"masks/{mask_name}_pct_above_0.95"] = 100 * (m_sig > 0.95).float().mean().item()
            
            wandb.log(log_dict)
    else:
        logger.info(f"Step {global_step}: training loss {outputs.loss.item():.4f}")
        
        if args.wandb:
            log_dict = {
                "train/loss": outputs.loss.item(),
                "global_step": global_step,
            }
            if epoch is not None:
                log_dict["epoch"] = epoch
            wandb.log(log_dict)

def init_layer_masks(masked_model, config, layer_idx, learn_head, learn_ffn_int, learn_ffn_out, hard_init):
    """Initialize masks for a specific layer."""
    if learn_head:
        if hard_init:
            masked_model.head_mask.data[layer_idx] = torch.ones(config.num_attention_heads).cuda() * 10
        else:
            masked_model.head_mask.data[layer_idx] = torch.randn(config.num_attention_heads).cuda()
    
    if learn_ffn_int:
        if hard_init:
            masked_model.ffn_intermediate_mask.data[layer_idx] = torch.ones(config.hidden_size).cuda() * 10
        else:
            masked_model.ffn_intermediate_mask.data[layer_idx] = torch.randn(config.hidden_size).cuda()
    
    if learn_ffn_out:
        if hard_init:
            masked_model.ffn_output_mask.data[layer_idx] = torch.ones(config.intermediate_size).cuda() * 10
        else:
            masked_model.ffn_output_mask.data[layer_idx] = torch.randn(config.intermediate_size).cuda()

def main():
    args = parser.parse_args()
    IS_SQUAD = "squad" in args.task_name
    IS_LARGE = "large" in args.model_name
    seq_len = 170 if IS_SQUAD else avg_seq_length(args.task_name)

    # Initialize wandb if enabled
    if args.wandb:
        import wandb
        wandb.init(
            entity="harvardml",
            project="td-pruning",
            name=f"{args.model_name}_{args.task_name}_{args.name}",
            config=vars(args),
        )
        logger.info(f"Weights & Biases logging enabled (project: td-pruning)")
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
    
    # Determine layer training order if layerwise training is enabled
    num_layers = config.num_hidden_layers
    if args.layerwise_train:
        if args.layerwise_order == 'front_to_back':
            layer_order = list(range(num_layers))
        elif args.layerwise_order == 'back_to_front':
            layer_order = list(range(num_layers - 1, -1, -1))
        elif args.layerwise_order == 'random':
            layer_order = list(range(num_layers))
            np.random.shuffle(layer_order)
        logger.info(f"Layer-wise training enabled with order: {args.layerwise_order}")
        logger.info(f"Training order: {layer_order}")
    else:
        layer_order = None

    if args.prune:
        # Determine which masks to learn
        learn_head = args.learn_masks in ['head', 'all']
        learn_ffn_int = args.learn_masks in ['ffn_int', 'all']
        learn_ffn_out = args.learn_masks in ['ffn_out', 'all']
        
        logger.info("Pruning enabled - using trainable masks")
        # For layerwise training, initialize as if hard_init is True (all masks start at ~1)
        effective_hard_init = args.hard_init or args.layerwise_train
        
        # Initialize masks: learned ones get random/hard init, fixed ones get 10 (sigmoid≈1)
        def init_mask(shape, learn):
            if learn:
                return torch.randn(*shape).cuda() if not effective_hard_init else torch.ones(*shape).cuda() * 10
            else:
                return torch.ones(*shape).cuda() * 10
        
        full_head_mask = init_mask((config.num_hidden_layers, config.num_attention_heads), learn_head)
        full_ffn_intermediate_mask = init_mask((config.num_hidden_layers, config.hidden_size), learn_ffn_int)
        full_ffn_output_mask = init_mask((config.num_hidden_layers, config.intermediate_size), learn_ffn_out)

        masked_model = make_masks_trainable(
            model,
            head_mask=full_head_mask,
            ffn_intermediate_mask=full_ffn_intermediate_mask,
            ffn_output_mask=full_ffn_output_mask,
        )

        # Set requires_grad based on which masks we're learning
        masked_model.head_mask.requires_grad = learn_head and not args.freeze
        masked_model.ffn_intermediate_mask.requires_grad = learn_ffn_int and not args.freeze
        masked_model.ffn_output_mask.requires_grad = learn_ffn_out and not args.freeze

        # Collect trainable mask parameters
        mask_params = [
            p for p in [masked_model.head_mask, masked_model.ffn_intermediate_mask, masked_model.ffn_output_mask]
            if p.requires_grad
        ]
        
        # All mask parameters (for exclusion from base_params)
        all_mask_params = [masked_model.head_mask, masked_model.ffn_intermediate_mask, masked_model.ffn_output_mask]
        mask_param_ids = {id(p) for p in all_mask_params}
        
        # Base params are everything except the masks
        base_params = [param for param in model.parameters() if id(param) not in mask_param_ids]
    else:
        logger.info("Pruning disabled - training without masks")
        mask_params = []
        base_params = [param for param in model.parameters() if param.requires_grad]

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
            {"params": base_params, "lr": 5e-5, "weight_decay": 0.01},
            {"params": mask_params, "lr": args.masks_LR, "weight_decay": 0.0},
        ]
    )
    
    # Learning rate scheduler with warmup
    # num_training_steps = args.num_epochs * len(masked_train_dataloader)
    # num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=num_training_steps
    # )
    
    model.train()
    global_step = 0

    if args.prune:
        m = model.trainable_ffn_intermediate_mask
        print("requires_grad:", m.requires_grad)
        ids = {id(p) for g in optimizer.param_groups for p in g['params']}
        print("in optimizer:", id(m) in ids)
    
    # Track a base model parameter to verify it's learning
    sample_param = None
    for name, param in model.named_parameters():
        if 'trainable' not in name and 'weight' in name and param.requires_grad:
            sample_param = (name, param.clone().detach())
            logger.info(f"Tracking param: {name}, initial mean: {param.data.mean().item():.6f}, std: {param.data.std().item():.6f}")
            break

    # Calculate total training steps for temperature annealing
    total_steps = args.num_epochs * len(masked_train_dataloader)
    
    # Start timing
    start = time.time()

    # Layer-wise training loop
    if args.layerwise_train and args.prune:
        logger.info(f"Starting layer-wise training for {num_layers} layers")
        # Calculate steps per layer: divide total epochs across all layers
        steps_per_layer = (args.num_epochs * len(masked_train_dataloader)) // num_layers
        batches_per_layer = steps_per_layer
        logger.info(f"Total steps: {total_steps}, Steps per layer: {steps_per_layer}")
        
        # Create an iterator over the training data
        train_iter = iter(masked_train_dataloader)
        
        for layer_idx_pos, layer_idx in enumerate(layer_order):
            logger.info(f"\n{'='*60}")
            logger.info(f"Training layer {layer_idx} ({layer_idx_pos + 1}/{num_layers})")
            logger.info(f"{'='*60}\n")
            
            # Reinitialize masks for this layer
            init_layer_masks(masked_model, config, layer_idx, learn_head, learn_ffn_int, learn_ffn_out, args.hard_init)
            
            # Train this layer for its allocated steps
            model.train()
            for step_in_layer in range(batches_per_layer):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    # Reset iterator if we run out of data
                    train_iter = iter(masked_train_dataloader)
                    batch = next(train_iter)
                
                loss, outputs, sparsity_loss, quantization_loss, current_temp = training_step(
                    batch, model, optimizer, args, global_step, total_steps, 
                    learn_head, learn_ffn_int, learn_ffn_out, layer_idx=layer_idx
                )
                global_step += 1

                if args.log_loss_every > 0 and global_step % args.log_loss_every == 0:
                    log_training_step(logger, wandb, args, global_step, outputs, loss, sparsity_loss, 
                                    quantization_loss, current_temp, model, learn_head, learn_ffn_int, 
                                    learn_ffn_out, epoch=None, layer_idx=layer_idx)
            
            # Evaluate after training this layer
            model.eval()
            
            head_mask_sigmoid = torch.sigmoid(masked_model.head_mask.detach())
            head_mask = (head_mask_sigmoid >= 0.5).float()
            
            ffn_intermediate_mask_sigmoid = torch.sigmoid(masked_model.ffn_intermediate_mask.detach())
            ffn_intermediate_mask = (ffn_intermediate_mask_sigmoid >= 0.5).float()
            
            ffn_output_mask_sigmoid = torch.sigmoid(masked_model.ffn_output_mask.detach())
            ffn_output_mask = (ffn_output_mask_sigmoid >= 0.5).float()
            
            epoch_test_acc = test_accuracy(model, head_mask, ffn_intermediate_mask, ffn_output_mask, tokenizer, args.task_name)
            logger.info(f"Layer {layer_idx} complete - Test accuracy: {epoch_test_acc:.4f}")
            
            if args.wandb:
                wandb.log({
                    "eval/accuracy": epoch_test_acc,
                    "global_step": global_step,
                    "current_layer": layer_idx,
                })
    else:
        # Standard training (all layers at once)
        for epoch in range(args.num_epochs):
            model.train()
            epoch_loss = 0.0
            for batch in masked_train_dataloader:
                loss, outputs, sparsity_loss, quantization_loss, current_temp = training_step(
                    batch, model, optimizer, args, global_step, total_steps, 
                    learn_head, learn_ffn_int, learn_ffn_out, layer_idx=None
                )
                epoch_loss += loss.item()
                global_step += 1

                if args.log_loss_every > 0 and global_step % args.log_loss_every == 0:
                    log_training_step(logger, wandb, args, global_step, outputs, loss, sparsity_loss, 
                                    quantization_loss, current_temp, model, learn_head, learn_ffn_int, 
                                    learn_ffn_out, epoch=epoch, layer_idx=None)

            # Evaluate at the end of each epoch
            model.eval()
            
            if args.prune:
                # Convert continuous masks to binary: sigmoid(mask) >= 0.5 -> 1, else -> 0
                head_mask_sigmoid = torch.sigmoid(masked_model.head_mask.detach())
                head_mask = (head_mask_sigmoid >= 0.5).float()
                
                ffn_intermediate_mask_sigmoid = torch.sigmoid(masked_model.ffn_intermediate_mask.detach())
                ffn_intermediate_mask = (ffn_intermediate_mask_sigmoid >= 0.5).float()
                
                ffn_output_mask_sigmoid = torch.sigmoid(masked_model.ffn_output_mask.detach())
                ffn_output_mask = (ffn_output_mask_sigmoid >= 0.5).float()
            else:
                head_mask = None
                ffn_intermediate_mask = None
                ffn_output_mask = None
            
            epoch_test_acc = test_accuracy(model, head_mask, ffn_intermediate_mask, ffn_output_mask, tokenizer, args.task_name)
            logger.info(f"Epoch {epoch + 1}/{args.num_epochs} - Test accuracy: {epoch_test_acc:.4f}")
            
            if args.wandb:
                wandb.log({
                    "eval/accuracy": epoch_test_acc,
                    "epoch": epoch,
                    "global_step": global_step,
                })
    
    end = time.time()
    logger.info(f"{args.task_name} Training time (s): {end - start}")
    
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
        torch.save(head_mask.cpu(), os.path.join(save_dir, "head_mask.pt"))
        torch.save(ffn_intermediate_mask.cpu(), os.path.join(save_dir, "ffn_intermediate_mask.pt"))
        torch.save(ffn_output_mask.cpu(), os.path.join(save_dir, "ffn_output_mask.pt"))
        logger.info(f"Masks saved to {save_dir}")

if __name__ == "__main__":
    main()
