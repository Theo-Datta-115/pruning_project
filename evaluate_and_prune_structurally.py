"""
Evaluate a saved pruned model and construct a structurally pruned version.

This script:
1. Loads a saved model and masks from the outputs directory
2. Evaluates the model with the saved masks (using hooks)
3. Constructs a smaller model by structurally removing zero-masked neurons
4. Evaluates the structurally pruned model

Usage:
    python evaluate_and_prune_structurally.py --save_name mnli_cos_cooldown2 --model_name bert-base-uncased --task_name mnli
"""

import argparse
import logging
import os
import copy

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    set_seed,
)

from evaluate_model.nlp import test_accuracy
from utils.model_loader import load_model_and_tokenizer
from utils.arch import get_layers, get_ffn1, get_ffn2, get_attn, get_mha_proj

logger = logging.getLogger(__name__)


def load_saved_model_and_masks(save_dir, model_name, task_name, use_base_model=False):
    """Load a saved model and its associated masks.
    
    Args:
        save_dir: Directory containing saved model and masks
        model_name: Base model name (e.g., bert-base-uncased)
        task_name: Task name (e.g., qqp, mnli)
        use_base_model: Whether to use base model architecture
        
    Returns:
        model, tokenizer, config, head_mask, ffn_mask
    """
    # Load the base model architecture
    default_root = "/n/netscratch/sham_lab/Everyone/tdatta/pruning/checkpoints/"
    config, model, tokenizer, model_source = load_model_and_tokenizer(
        model_name=model_name,
        task_name=task_name,
        ckpt_dir=None,
        use_base_model=use_base_model,
        default_root=default_root,
    )
    
    # Load the saved weights
    weights_path = os.path.join(save_dir, "model_weights.pt")
    if os.path.exists(weights_path):
        logger.info(f"Loading model weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Remove trainable mask parameters from state dict (they're not part of the base model)
        state_dict = {k: v for k, v in state_dict.items() 
                     if not k.startswith('trainable_')}
        
        model.load_state_dict(state_dict, strict=False)
    else:
        logger.warning(f"No model weights found at {weights_path}, using base model")
    
    # Load masks
    head_mask_path = os.path.join(save_dir, "head_mask.pt")
    ffn_mask_path = os.path.join(save_dir, "ffn_mask.pt")
    
    head_mask = torch.load(head_mask_path) if os.path.exists(head_mask_path) else None
    ffn_mask = torch.load(ffn_mask_path) if os.path.exists(ffn_mask_path) else None
    
    logger.info(f"Loaded masks:")
    logger.info(f"  Head mask: {head_mask.shape if head_mask is not None else 'None'}")
    logger.info(f"  FFN mask: {ffn_mask.shape if ffn_mask is not None else 'None'}")
    
    return model, tokenizer, config, head_mask, ffn_mask


def create_structurally_pruned_model(model, config, head_mask, ffn_mask):
    """Create a smaller model by structurally removing zero-masked components.
    
    This function creates a new model with reduced dimensions based on the masks.
    - For attention heads: removes entire heads that are masked out (mask value = 0)
    - For FFN layers: removes intermediate neurons that are masked out (mask value = 0)
    
    Args:
        model: Original model
        config: Model configuration
        head_mask: Head attention mask [num_layers, num_heads] (optional)
        ffn_mask: FFN mask [num_layers, intermediate_size] (optional)
        
    Returns:
        pruned_model: New model with reduced structure
        new_config: Updated configuration
    """
    logger.info("Creating structurally pruned model...")
    
    # Create a new config with potentially modified dimensions
    new_config = copy.deepcopy(config)
    
    # Create a new model with the same config (we'll manually prune it)
    is_squad = (hasattr(model.config, 'task_specific_params') and 
               model.config.task_specific_params is not None and 
               any("squad" in k.lower() for k in model.config.task_specific_params.keys()))
    model_cls = type(model)
    pruned_model = model_cls(new_config)
    
    # Copy the base model weights
    pruned_model.load_state_dict(model.state_dict(), strict=False)
    
    # Get layers for both models
    layers = get_layers(model)
    pruned_layers = get_layers(pruned_model)
    
    # Prune both attention heads and FFN layers
    for layer_idx in range(len(layers)):
        logger.info(f"Pruning layer {layer_idx}...")
        
        # ===== ATTENTION HEAD PRUNING =====
        if head_mask is not None:
            head_layer_mask = head_mask[layer_idx]  # [num_heads]
            head_kept_indices = (head_layer_mask > 0).nonzero(as_tuple=True)[0]
            
            # Handle edge case: if all heads are pruned, keep at least 1
            if len(head_kept_indices) == 0:
                logger.warning(f"  Layer {layer_idx}: All attention heads pruned! Keeping 1 head to avoid breaking model.")
                head_kept_indices = torch.tensor([0])
            
            # Get attention modules
            orig_attn = get_attn(model, layer_idx)  # self-attention (Q, K, V)
            orig_mha_proj = get_mha_proj(model, layer_idx)  # output projection
            pruned_attn = get_attn(pruned_model, layer_idx)
            pruned_mha_proj = get_mha_proj(pruned_model, layer_idx)
            
            # Get dimensions
            num_heads = config.num_attention_heads
            head_dim = config.hidden_size // num_heads
            hidden_size = config.hidden_size
            
            # Prune Q, K, V projections (each is [hidden_size, hidden_size])
            # They are concatenated, so we need to select specific head slices
            for proj_name in ['query', 'key', 'value']:
                orig_proj = getattr(orig_attn, proj_name)
                pruned_proj = getattr(pruned_attn, proj_name)
                
                # Original weight: [hidden_size, hidden_size]
                # Reshaped: [num_heads, head_dim, hidden_size]
                orig_weight = orig_proj.weight.data.view(num_heads, head_dim, hidden_size)
                orig_bias = orig_proj.bias.data.view(num_heads, head_dim) if orig_proj.bias is not None else None
                
                # Keep only selected heads
                new_weight = orig_weight[head_kept_indices].reshape(-1, hidden_size)
                new_bias = orig_bias[head_kept_indices].reshape(-1) if orig_bias is not None else None
                
                # Create new linear layer with reduced dimensions
                new_proj = nn.Linear(hidden_size, len(head_kept_indices) * head_dim, bias=(orig_bias is not None))
                new_proj.weight.data = new_weight
                if new_bias is not None:
                    new_proj.bias.data = new_bias
                
                setattr(pruned_attn, proj_name, new_proj)
            
            # Prune output projection: [hidden_size, hidden_size]
            # Input dimension changes (concatenated heads), output stays the same
            orig_out_weight = orig_mha_proj.dense.weight.data  # [hidden_size, hidden_size]
            orig_out_bias = orig_mha_proj.dense.bias.data if orig_mha_proj.dense.bias is not None else None
            
            # Reshape to separate heads: [hidden_size, num_heads, head_dim]
            orig_out_weight = orig_out_weight.view(hidden_size, num_heads, head_dim)
            # Keep only selected heads
            new_out_weight = orig_out_weight[:, head_kept_indices, :].reshape(hidden_size, -1)
            
            new_out_proj = nn.Linear(len(head_kept_indices) * head_dim, hidden_size, 
                                     bias=(orig_out_bias is not None))
            new_out_proj.weight.data = new_out_weight
            if orig_out_bias is not None:
                new_out_proj.bias.data = orig_out_bias
            
            pruned_mha_proj.dense = new_out_proj
            
            # Update num_attention_heads so transpose_for_scores works correctly
            pruned_attn.num_attention_heads = len(head_kept_indices)
            # Also update all_head_size for consistency
            pruned_attn.all_head_size = len(head_kept_indices) * head_dim
            
            head_sparsity = 1 - len(head_kept_indices) / num_heads
            logger.info(f"  Layer {layer_idx}: Attention heads {num_heads} -> "
                       f"{len(head_kept_indices)} (sparsity: {head_sparsity:.1%})")
        
        # ===== FFN PRUNING =====
        # Get the original layer modules
        orig_ffn1 = get_ffn1(model, layer_idx)  # intermediate layer
        orig_ffn2 = get_ffn2(model, layer_idx)  # output layer
        
        pruned_ffn1 = get_ffn1(pruned_model, layer_idx)
        pruned_ffn2 = get_ffn2(pruned_model, layer_idx)
        
        # Get the original weights
        orig_ffn1_weight = orig_ffn1.dense.weight.data  # [intermediate_size, hidden_size]
        orig_ffn1_bias = orig_ffn1.dense.bias.data if orig_ffn1.dense.bias is not None else None
        orig_ffn2_weight = orig_ffn2.dense.weight.data  # [hidden_size, intermediate_size]
        orig_ffn2_bias = orig_ffn2.dense.bias.data if orig_ffn2.dense.bias is not None else None
        
        # FFN mask: which intermediate neurons to keep
        if ffn_mask is not None:
            intermediate_mask = ffn_mask[layer_idx]  # [intermediate_size]
            intermediate_kept_indices = (intermediate_mask > 0).nonzero(as_tuple=True)[0]
        else:
            intermediate_kept_indices = torch.arange(orig_ffn1_weight.shape[0])
        
        # Handle edge case: if all neurons are pruned, keep at least 1 to avoid breaking the model
        if len(intermediate_kept_indices) == 0:
            logger.warning(f"  Layer {layer_idx}: All intermediate neurons pruned! Keeping 1 neuron to avoid breaking model.")
            intermediate_kept_indices = torch.tensor([0])
        
        # Prune FFN1: input_dim (hidden_size) -> output_dim (intermediate_size)
        # Keep only: selected output rows (intermediate neurons)
        # Input dimension stays the same (hidden_size)
        new_ffn1_weight = orig_ffn1_weight[intermediate_kept_indices, :]
        new_ffn1_bias = orig_ffn1_bias[intermediate_kept_indices] if orig_ffn1_bias is not None else None
        
        new_ffn1_dense = nn.Linear(orig_ffn1_weight.shape[1], len(intermediate_kept_indices), 
                                   bias=(orig_ffn1_bias is not None))
        new_ffn1_dense.weight.data = new_ffn1_weight
        if new_ffn1_bias is not None:
            new_ffn1_dense.bias.data = new_ffn1_bias
        
        pruned_ffn1.dense = new_ffn1_dense
        
        # Prune FFN2: input_dim (intermediate_size) -> output_dim (hidden_size)
        # Keep only: selected input columns (intermediate neurons)
        new_ffn2_weight = orig_ffn2_weight[:, intermediate_kept_indices]
        new_ffn2_bias = orig_ffn2_bias  # Output dimension doesn't change
        
        new_ffn2_dense = nn.Linear(len(intermediate_kept_indices), orig_ffn2_weight.shape[0],
                                   bias=(orig_ffn2_bias is not None))
        new_ffn2_dense.weight.data = new_ffn2_weight
        if new_ffn2_bias is not None:
            new_ffn2_dense.bias.data = new_ffn2_bias
        
        pruned_ffn2.dense = new_ffn2_dense
        
        sparsity = 1 - len(intermediate_kept_indices) / orig_ffn1_weight.shape[0]
        logger.info(f"  Layer {layer_idx}: FFN intermediate neurons {orig_ffn1_weight.shape[0]} -> "
                   f"{len(intermediate_kept_indices)} (sparsity: {sparsity:.1%})")
    
    # Count parameters
    orig_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    logger.info(f"Original model parameters: {orig_params:,}")
    logger.info(f"Pruned model parameters: {pruned_params:,}")
    logger.info(f"Parameter reduction: {(1 - pruned_params/orig_params)*100:.2f}%")
    
    return pruned_model, new_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate and structurally prune a saved model")
    parser.add_argument("--save_name", type=str, required=True, 
                       help="Name of the saved model directory (inside outputs/)")
    parser.add_argument("--model_name", type=str, required=True,
                       help="Base model name (e.g., bert-base-uncased)")
    parser.add_argument("--task_name", type=str, required=True,
                       help="Task name (e.g., qqp, mnli, squad)")
    parser.add_argument("--output_dir", type=str, 
                       default="/n/netscratch/sham_lab/Everyone/tdatta/pruning/outputs/",
                       help="Base output directory")
    parser.add_argument("--use_base_model", action="store_true",
                       help="Load base model architecture instead of fine-tuned")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Set GPU and seed
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)
    
    # Construct the save directory path
    save_dir = os.path.join(args.output_dir, args.save_name)
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Save directory not found: {save_dir}")
    
    logger.info(f"Loading model from {save_dir}")
    
    # Load the saved model and masks
    model, tokenizer, config, head_mask, ffn_mask = load_saved_model_and_masks(
        save_dir, args.model_name, args.task_name, args.use_base_model
    )
    
    model = model.cuda()
    model.eval()
    
    # Move masks to GPU
    if head_mask is not None:
        head_mask = head_mask.cuda()
    if ffn_mask is not None:
        ffn_mask = ffn_mask.cuda()
    
    # Evaluate the original model with masks
    logger.info("\n" + "="*80)
    logger.info("Evaluating original model with masks...")
    logger.info("="*80)
    
    masked_accuracy = test_accuracy(
        model, head_mask, ffn_mask,
        tokenizer, args.task_name
    )
    logger.info(f"Masked model accuracy: {masked_accuracy:.4f}")
    
    # Create structurally pruned model
    logger.info("\n" + "="*80)
    logger.info("Creating structurally pruned model...")
    logger.info("="*80)
    
    pruned_model, pruned_config = create_structurally_pruned_model(
        model, config, head_mask, ffn_mask
    )
    
    pruned_model = pruned_model.cuda()
    pruned_model.eval()
    
    # Evaluate the structurally pruned model (no masks needed)
    logger.info("\n" + "="*80)
    logger.info("Evaluating structurally pruned model...")
    logger.info("="*80)
    
    pruned_accuracy = test_accuracy(
        pruned_model, None, None,
        tokenizer, args.task_name
    )
    logger.info(f"Structurally pruned model accuracy: {pruned_accuracy:.4f}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Masked model accuracy:            {masked_accuracy:.4f}")
    logger.info(f"Structurally pruned model accuracy: {pruned_accuracy:.4f}")
    logger.info(f"Accuracy difference:               {abs(masked_accuracy - pruned_accuracy):.4f}")
    
    # Save the structurally pruned model
    pruned_save_dir = os.path.join(save_dir, "structurally_pruned")
    os.makedirs(pruned_save_dir, exist_ok=True)
    
    torch.save(pruned_model.state_dict(), os.path.join(pruned_save_dir, "model_weights.pt"))
    pruned_config.save_pretrained(pruned_save_dir)
    
    logger.info(f"\nStructurally pruned model saved to {pruned_save_dir}")


if __name__ == "__main__":
    main()
