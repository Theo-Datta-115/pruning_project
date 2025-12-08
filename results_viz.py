"""
Visualize pruning mask distributions from saved model checkpoints.

Usage:
    python results_viz.py --save_path /n/netscratch/sham_lab/Everyone/tdatta/pruning/outputs/mnli_cos_cooldown2 --model_name bert-base-uncased --task_name mnli
"""

import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoConfig

from utils.model_loader import load_model_and_tokenizer
from utils.arch import get_attention_param_ratio


def plot_pruning_per_layer(head_mask, ffn_mask, attention_param_ratio, save_dir, save_name):
    """Plot the percentage of parameters pruned per layer for heads, FFN, and total."""
    num_layers = head_mask.shape[0]
    layers = np.arange(num_layers)
    
    # Calculate pruning percentages (percentage of parameters kept, using 0.5 threshold)
    head_kept_pct = ((head_mask >= 0.5).float().mean(dim=1) * 100).cpu().numpy()
    ffn_kept_pct = ((ffn_mask >= 0.5).float().mean(dim=1) * 100).cpu().numpy()
    
    # Calculate total pruning per layer (weighted by parameter count)
    # attention_param_ratio = attention_params / ffn_params
    # Total params in layer = attention_params + ffn_params
    # Weight for attention = attention_params / (attention_params + ffn_params)
    # Weight for FFN = ffn_params / (attention_params + ffn_params)
    attn_weight = attention_param_ratio / (attention_param_ratio + 1)
    ffn_weight = 1 / (attention_param_ratio + 1)
    
    total_kept_pct = (attn_weight * head_kept_pct + ffn_weight * ffn_kept_pct)
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Head pruning
    axes[0].bar(layers, head_kept_pct, color='steelblue', alpha=0.8)
    # axes[0].axhline(y=head_kept_pct.mean(), color='red', linestyle='--', 
                    # label=f'Mean: {head_kept_pct.mean():.1f}%')
    axes[0].set_xlabel('Layer', fontsize=12)
    axes[0].set_ylabel('% Parameters Kept', fontsize=12)
    axes[0].set_title('Attention Head Pruning per Layer', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 105])
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # FFN pruning
    axes[1].bar(layers, ffn_kept_pct, color='darkorange', alpha=0.8)
    # axes[1].axhline(y=ffn_kept_pct.mean(), color='red', linestyle='--', 
    #                 label=f'Mean: {ffn_kept_pct.mean():.1f}%')
    axes[1].set_xlabel('Layer', fontsize=12)
    axes[1].set_ylabel('% Parameters Kept', fontsize=12)
    axes[1].set_title('FFN Pruning per Layer', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 105])
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # Total pruning (weighted)
    axes[2].bar(layers, total_kept_pct, color='forestgreen', alpha=0.8)
    # axes[2].axhline(y=total_kept_pct.mean(), color='red', linestyle='--', 
    #                 label=f'Mean: {total_kept_pct.mean():.1f}%')
    axes[2].set_xlabel('Layer', fontsize=12)
    axes[2].set_ylabel('% Parameters Kept', fontsize=12)
    axes[2].set_title('Total Pruning per Layer (Weighted)', fontsize=14, fontweight='bold')
    axes[2].set_ylim([0, 105])
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{save_name}_pruning_per_layer.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_mask_value_distributions(head_mask, ffn_mask, save_dir, save_name):
    """Plot histograms of mask values for attention heads and FFN."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Attention head mask distribution
    head_values = head_mask.cpu().numpy().flatten()
    axes[0].hist(head_values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
    axes[0].set_xlabel('Mask Value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Attention Head Mask Value Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add statistics
    below_threshold = (head_values < 0.5).sum()
    total = len(head_values)
    axes[0].text(0.02, 0.98, f'Below 0.5: {below_threshold}/{total} ({100*below_threshold/total:.1f}%)',
                 transform=axes[0].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # FFN mask distribution
    ffn_values = ffn_mask.cpu().numpy().flatten()
    axes[1].hist(ffn_values, bins=50, color='darkorange', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
    axes[1].set_xlabel('Mask Value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('FFN Mask Value Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add statistics
    below_threshold = (ffn_values < 0.5).sum()
    total = len(ffn_values)
    axes[1].text(0.02, 0.98, f'Below 0.5: {below_threshold}/{total} ({100*below_threshold/total:.1f}%)',
                 transform=axes[1].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{save_name}_mask_distributions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize pruning mask distributions')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to directory containing head_mask.pt and ffn_mask.pt')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name (e.g., bert-base-uncased) for loading config')
    parser.add_argument('--task_name', type=str, required=True,
                        help='Task name (e.g., qqp, mnli)')
    parser.add_argument('--output_dir', type=str, default='images',
                        help='Directory to save visualization images')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load masks
    head_mask_path = os.path.join(args.save_path, 'head_mask.pt')
    ffn_mask_path = os.path.join(args.save_path, 'ffn_mask.pt')
    
    if not os.path.exists(head_mask_path):
        raise FileNotFoundError(f"Head mask not found: {head_mask_path}")
    if not os.path.exists(ffn_mask_path):
        raise FileNotFoundError(f"FFN mask not found: {ffn_mask_path}")
    
    head_mask = torch.load(head_mask_path)
    ffn_mask = torch.load(ffn_mask_path)
    
    print(f"Loaded masks from: {args.save_path}")
    print(f"Head mask shape: {head_mask.shape}")
    print(f"FFN mask shape: {ffn_mask.shape}")
    
    # Load model to get attention parameter ratio
    default_root = "/n/netscratch/sham_lab/Everyone/tdatta/pruning/checkpoints/"
    config, model, tokenizer, model_source = load_model_and_tokenizer(
        model_name=args.model_name,
        task_name=args.task_name,
        ckpt_dir=None,
        use_base_model=True,
        default_root=default_root,
    )
    
    attention_param_ratio = get_attention_param_ratio(model)
    print(f"Attention to FFN parameter ratio: {attention_param_ratio:.4f}")
    
    # Generate save name from path
    save_name = os.path.basename(args.save_path.rstrip('/'))
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_pruning_per_layer(head_mask, ffn_mask, attention_param_ratio, args.output_dir, save_name)
    plot_mask_value_distributions(head_mask, ffn_mask, args.output_dir, save_name)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Attention Heads:")
    print(f"  - Total parameters: {head_mask.numel()}")
    head_kept = (head_mask >= 0.5).sum().item()
    head_pruned = (head_mask < 0.5).sum().item()
    print(f"  - Parameters kept: {head_kept:.0f} ({100*head_kept/head_mask.numel():.2f}%)")
    print(f"  - Parameters pruned: {head_pruned:.0f} ({100*head_pruned/head_mask.numel():.2f}%)")
    
    print(f"\nFFN Neurons:")
    print(f"  - Total parameters: {ffn_mask.numel()}")
    ffn_kept = (ffn_mask >= 0.5).sum().item()
    ffn_pruned = (ffn_mask < 0.5).sum().item()
    print(f"  - Parameters kept: {ffn_kept:.0f} ({100*ffn_kept/ffn_mask.numel():.2f}%)")
    print(f"  - Parameters pruned: {ffn_pruned:.0f} ({100*ffn_pruned/ffn_mask.numel():.2f}%)")
    
    # Calculate weighted total
    attn_weight = attention_param_ratio / (attention_param_ratio + 1)
    ffn_weight = 1 / (attention_param_ratio + 1)
    total_kept_pct = attn_weight * (head_kept/head_mask.numel()) + ffn_weight * (ffn_kept/ffn_mask.numel())
    
    print(f"\nWeighted Total:")
    print(f"  - Parameters kept: {100*total_kept_pct:.2f}%")
    print(f"  - Parameters pruned: {100*(1-total_kept_pct):.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()
