# Transformer Pruning with Trainable Masks

Train neural network pruning masks using Gumbel-Sigmoid relaxation. This framework learns which attention heads and FFN neurons to prune during training, using differentiable masks with temperature annealing.

## Quick Start

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

Requires Python 3.7+ and an NVIDIA GPU with 16+ GB memory.

### Download Checkpoints

Use the helper script to download pre-trained models:

```bash
python scripts/download_checkpoints.py --tasks qqp --models bert-base-uncased
```

This creates `checkpoints/bert-base-uncased/qqp/` with the model files needed by `main.py`.

## Training with Learnable Masks

`main.py` trains a model with learnable pruning masks using Gumbel-Sigmoid:

```bash
python main.py --model_name bert-base-uncased \
               --task_name qqp \
               --prune \
               --learn_masks ffn_int \
               --num_epochs 5 \
               --wandb
```

### Key Arguments

**Required:**
- `--model_name`: Model architecture (e.g., `bert-base-uncased`)
- `--task_name`: Task to train on (`qqp`, `mnli`, `qnli`, `sst2`, `mrpc`, `stsb`, `squad`, `squad_v2`)
- `--prune`: Enable trainable mask learning

**Mask Selection:**
- `--learn_masks`: Which masks to learn (`head`, `ffn_int`, `ffn_out`, `all`)

**Gumbel-Sigmoid Temperature:**
- `--gumbel_temp_start`: Starting temperature (default: 5.0, higher = more exploration)
- `--gumbel_temp_end`: Ending temperature (default: 0.1, lower = more discrete)
- `--gumbel_temp_anneal`: Schedule (`linear`, `exponential`, `constant`)

**Loss Weights:**
- `--sparsity_loss`: Weight for sparsity regularization (default: 0.1)
- `--quantization_loss`: Weight for binary quantization (default: 0.01)

**Training:**
- `--num_epochs`: Number of training epochs (default: 5)
- `--masks_LR`: Learning rate for masks (default: 0.01)
- `--freeze`: Freeze model weights, only train masks

**Logging:**
- `--wandb`: Enable Weights & Biases logging
- `--name`: Run name for wandb
- `--log_loss_every`: Log frequency in steps (default: 5)

### Output

Trained masks are saved to `--output_dir` as:
- `head_mask.pt`
- `ffn_intermediate_mask.pt`
- `ffn_output_mask.pt`

## What to Change for Your Setup

1. **Checkpoint directory**: Update `--output_dir` in `main.py` (line 76) or pass via command line
2. **Default checkpoint root**: Change `default_root` in `scripts/download_checkpoints.py` (line 78)
3. **Downloading Models**: Use `scripts/download_checkpoints.py` to download models from Hugging Face Hub
4. **Changing Wandb Logging**: Use `--wandb` to enable Weights & Biases logging, and change target directory in `main.py` (line 76)

## How It Works

Trains models with learnable pruning masks:
- Wraps model with trainable mask parameters
- Applies Gumbel-Sigmoid during forward pass for differentiable sampling
- Anneals temperature from high (exploration) to low (discrete decisions)
- Computes sparsity (mean) and quantization (binary cross-entropy) losses to encourage binary masks
- Saves final binary masks after training

## Supported Models & Tasks

**Models:** BERT-base/large, DistilBERT, RoBERTa-base/large, etc.

**Tasks:**
- GLUE: MNLI, QQP, QNLI, SST-2, STS-B, MRPC
- SQuAD V1.1 & V2
