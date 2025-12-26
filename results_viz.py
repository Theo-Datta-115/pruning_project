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


def plot_pruning_per_layer(head_mask, ffn_mask, save_dir, save_name):
    """Plot the percentage of parameters pruned per layer for heads and FFN."""
    num_layers = head_mask.shape[0]
    layers = np.arange(num_layers)
    
    # Calculate pruning percentages (percentage of parameters kept, using 0.5 threshold)
    head_kept_pct = ((head_mask >= 0.5).float().mean(dim=1) * 100).cpu().numpy()
    ffn_kept_pct = ((ffn_mask >= 0.5).float().mean(dim=1) * 100).cpu().numpy()
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Head pruning
    axes[0].bar(layers, head_kept_pct, color='steelblue', alpha=0.8)
    axes[0].set_xlabel('Layer', fontsize=12)
    axes[0].set_ylabel('% Parameters Kept', fontsize=12)
    axes[0].set_title('Attention Head Pruning per Layer', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 105])
    axes[0].grid(axis='y', alpha=0.3)
    
    # FFN pruning
    axes[1].bar(layers, ffn_kept_pct, color='darkorange', alpha=0.8)
    axes[1].set_xlabel('Layer', fontsize=12)
    axes[1].set_ylabel('% Parameters Kept', fontsize=12)
    axes[1].set_title('FFN Pruning per Layer', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 105])
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{save_name}_pruning_per_layer.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_pruning_comparison(model_names, save_dir):
    """Plot per-layer pruning comparison for multiple models."""
    outputs_path = "/n/netscratch/sham_lab/Everyone/tdatta/pruning/outputs"
    
    # Hard-coded display names for comparison
    display_names = ['Multiindex', 'Polynomial']
    
    # Load masks for each model
    masks_data = []
    for i, name in enumerate(model_names):
        model_path = os.path.join(outputs_path, name)
        head_mask = torch.load(os.path.join(model_path, 'head_mask.pt'))
        ffn_mask = torch.load(os.path.join(model_path, 'ffn_mask.pt'))
        masks_data.append({
            'name': display_names[i] if i < len(display_names) else name,
            'head_kept_pct': ((head_mask >= 0.5).float().mean(dim=1) * 100).cpu().numpy(),
            'ffn_kept_pct': ((ffn_mask >= 0.5).float().mean(dim=1) * 100).cpu().numpy(),
        })
    
    num_layers = len(masks_data[0]['head_kept_pct'])
    layers = np.arange(num_layers)
    num_models = len(model_names)
    
    # Bar width and offsets for grouped bars
    bar_width = 0.8 / num_models
    colors = ['steelblue', 'coral', 'seagreen', 'orchid', 'goldenrod']
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Head pruning comparison
    for i, data in enumerate(masks_data):
        offset = (i - num_models / 2 + 0.5) * bar_width
        axes[0].bar(layers + offset, data['head_kept_pct'], bar_width, 
                    color=colors[i % len(colors)], alpha=0.8, label=data['name'])
    axes[0].set_xlabel('Layer', fontsize=12)
    axes[0].set_ylabel('% Parameters Kept', fontsize=12)
    axes[0].set_title('Attention Head Pruning per Layer', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 105])
    axes[0].set_xticks(layers)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # FFN pruning comparison
    for i, data in enumerate(masks_data):
        offset = (i - num_models / 2 + 0.5) * bar_width
        axes[1].bar(layers + offset, data['ffn_kept_pct'], bar_width,
                    color=colors[i % len(colors)], alpha=0.8, label=data['name'])
    axes[1].set_xlabel('Layer', fontsize=12)
    axes[1].set_ylabel('% Parameters Kept', fontsize=12)
    axes[1].set_title('FFN Pruning per Layer', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 105])
    axes[1].set_xticks(layers)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_name = '_vs_'.join(model_names)
    save_path = os.path.join(save_dir, f'{save_name}_comparison.png')
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


# Default outputs path
OUTPUTS_PATH = "/n/netscratch/sham_lab/Everyone/tdatta/pruning/outputs"


def main():
    parser = argparse.ArgumentParser(description='Visualize pruning mask distributions')
    parser.add_argument('--save_name', type=str, nargs='+', required=True,
                        help='Model save name(s) in outputs directory. If 2+ names provided, generates comparison plot.')
    parser.add_argument('--output_dir', type=str, default='images',
                        help='Directory to save visualization images')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If multiple models, generate comparison plot
    if len(args.save_name) >= 2:
        print(f"Generating comparison plot for: {args.save_name}")
        plot_pruning_comparison(args.save_name, args.output_dir)
        return
    
    # Single model case
    save_name = args.save_name[0]
    save_path = os.path.join(OUTPUTS_PATH, save_name)
    
    # Load masks
    head_mask_path = os.path.join(save_path, 'head_mask.pt')
    ffn_mask_path = os.path.join(save_path, 'ffn_mask.pt')
    
    if not os.path.exists(head_mask_path):
        raise FileNotFoundError(f"Head mask not found: {head_mask_path}")
    if not os.path.exists(ffn_mask_path):
        raise FileNotFoundError(f"FFN mask not found: {ffn_mask_path}")
    
    head_mask = torch.load(head_mask_path)
    ffn_mask = torch.load(ffn_mask_path)
    
    print(f"Loaded masks from: {save_path}")
    print(f"Head mask shape: {head_mask.shape}")
    print(f"FFN mask shape: {ffn_mask.shape}")
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_pruning_per_layer(head_mask, ffn_mask, args.output_dir, save_name)
    plot_mask_value_distributions(head_mask, ffn_mask, args.output_dir, save_name)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Attention Heads:")
    print(f"  - Total: {head_mask.numel()}")
    head_kept = (head_mask >= 0.5).sum().item()
    head_pruned = (head_mask < 0.5).sum().item()
    print(f"  - Kept: {head_kept:.0f} ({100*head_kept/head_mask.numel():.2f}%)")
    print(f"  - Pruned: {head_pruned:.0f} ({100*head_pruned/head_mask.numel():.2f}%)")
    
    print(f"\nFFN Neurons:")
    print(f"  - Total: {ffn_mask.numel()}")
    ffn_kept = (ffn_mask >= 0.5).sum().item()
    ffn_pruned = (ffn_mask < 0.5).sum().item()
    print(f"  - Kept: {ffn_kept:.0f} ({100*ffn_kept/ffn_mask.numel():.2f}%)")
    print(f"  - Pruned: {ffn_pruned:.0f} ({100*ffn_pruned/ffn_mask.numel():.2f}%)")
    print("="*60)


if __name__ == '__main__':
    main()
