"""
Synthetic datasets for testing MLP and MHA learning capabilities with BERT.

1. PolynomialDataset: f(x) = sin(5x) + x^3 - 0.2x^4
   - Tests MLP learning (function approximation)
   - Input: integer x as text, Output: f(x) as regression target

2. IndexSumDataset: output = seq[seq[0]] + seq[seq[1]] + seq[seq[2]]
   - Tests MHA learning (attention/indexing)
   - Input: sequence of integers as text, Output: sum of values at indexed positions

Both datasets use text-encoded integers for BERT compatibility.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class PolynomialDataset(Dataset):
    """
    Dataset for polynomial function: f(x) = sin(5x) + x^3 - 0.2x^4 + noise
    
    Uses continuous float x values in range [-2, 2] passed directly to model.
    Bypasses tokenization - model uses input_floats instead of input_ids.
    Adds Gaussian noise to outputs to prevent memorization.
    Designed to test MLP function approximation capabilities.
    """
    
    def __init__(self, num_samples: int = 100000, seed: int = 42, noise_std: float = 0.1):
        """
        Args:
            num_samples: Number of examples in the dataset
            seed: Random seed for reproducibility
            noise_std: Standard deviation of Gaussian noise added to outputs
        """
        self.num_samples = num_samples
        
        rng = np.random.default_rng(seed)
        
        # Generate continuous x values in [-2, 2]
        self.x_values = rng.uniform(-2.0, 2.0, size=num_samples).astype(np.float32)
        
        # Compute f(x) = sin(5x) + x^3 - 0.2*x^4
        y_clean = np.where(self.x_values < -1, self.x_values**2, 
        np.where(self.x_values < 0, np.sin(10*self.x_values),
        np.where(self.x_values < 1, np.exp(self.x_values) - 1, 
        np.log(self.x_values + 1) + 1)))
        # y_clean = (
        #     np.sin(5 * self.x_values) 
        #     + self.x_values ** 3 - 
        #     0.2 * self.x_values ** 4
        # )
        
        # Add Gaussian noise to prevent memorization
        noise = rng.normal(0, noise_std, size=num_samples)
        self.y_values = (y_clean + noise).astype(np.float32)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        x = self.x_values[idx]
        y = self.y_values[idx]
        
        # Return raw float - model handles embedding via input_floats
        return {
            "input_floats": torch.tensor([x], dtype=torch.float32),
            "labels": torch.tensor([y], dtype=torch.float32),
        }


class IndexSumDataset(Dataset):
    """
    Dataset where output = seq[seq[0]] + seq[seq[1]] + seq[seq[2]]
    
    The first 3 values of the sequence are indices pointing to later positions.
    The output is the sum of values at those indexed positions.
    All values are small integers encoded as text for BERT.
    
    Designed to test MHA's ability to learn attention/indexing patterns.
    """
    
    def __init__(self, tokenizer, num_samples: int = 100000, seq_len: int = 16, 
                 seed: int = 42, max_seq_len: int = 64):
        """
        Args:
            tokenizer: BERT tokenizer for encoding text
            num_samples: Number of examples in the dataset
            seq_len: Number of integers in the sequence
            seed: Random seed for reproducibility
            max_seq_len: Maximum sequence length for tokenization
        """
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.max_seq_len = max_seq_len
        
        rng = np.random.default_rng(seed)
        
        # Generate sequences of small integers
        # First 3 positions: indices (3 to seq_len-1) pointing to later positions
        # Remaining positions: random integers in [-20, 20]
        self.sequences = np.zeros((num_samples, seq_len), dtype=np.int32)
        
        # First 3 elements are indices pointing to positions 3+
        self.sequences[:, 0] = rng.integers(3, seq_len, size=num_samples)
        self.sequences[:, 1] = rng.integers(3, seq_len, size=num_samples)
        self.sequences[:, 2] = rng.integers(3, seq_len, size=num_samples)
        
        # Fill remaining positions with random integers
        self.sequences[:, 3:] = rng.integers(-20, 21, size=(num_samples, seq_len - 3))
        
        # Compute targets: sum of values at the indexed positions
        self.targets = np.zeros(num_samples, dtype=np.float32)
        for i in range(num_samples):
            idx0 = self.sequences[i, 0]
            idx1 = self.sequences[i, 1]
            idx2 = self.sequences[i, 2]
            self.targets[i] = (
                self.sequences[i, idx0] + 
                self.sequences[i, idx1] + 
                self.sequences[i, idx2]
            )
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target = self.targets[idx]
        
        # Text-encode: space-separated integers
        # Format: "4 2 6 | 3 -5 12 8 -1 0 7 15 -3 2 9 -8 4"
        # The | separates indices from values for clarity
        indices_str = " ".join(str(x) for x in seq[:3])
        values_str = " ".join(str(x) for x in seq[3:])
        text = f"{indices_str} | {values_str}"
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor([target], dtype=torch.float32),
        }


class MultiIndexDataset(Dataset):
    """
    Dataset where output = seq[seq[...seq[0]...]] (n_hops nested index lookups).
    
    All values are integers in [0, seq_len-1], making this a classification task
    with seq_len classes. Requires attention to follow the chain of indices.
    
    Designed to test MHA's ability to learn multi-hop attention patterns.
    """
    
    def __init__(self, tokenizer, num_samples: int = 100000, seq_len: int = 16, 
                 seed: int = 42, max_seq_len: int = 64, n_hops: int = 16):
        """
        Args:
            tokenizer: BERT tokenizer for encoding text
            num_samples: Number of examples in the dataset
            seq_len: Number of integers in the sequence (also = num_classes)
            seed: Random seed for reproducibility
            max_seq_len: Maximum sequence length for tokenization
            n_hops: Number of index lookups to perform. Defaults to seq_len.
        """
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_classes = seq_len
        self.max_seq_len = max_seq_len
        self.n_hops = n_hops 
        
        rng = np.random.default_rng(seed)
        
        # Generate sequences where all values are valid indices [0, seq_len-1]
        self.sequences = rng.integers(0, seq_len, size=(num_samples, seq_len), dtype=np.int32)
        
        # Compute targets: n_hops nested lookups starting from index 0
        self.targets = np.zeros(num_samples, dtype=np.int64)
        for i in range(num_samples):
            idx = 0
            for _ in range(self.n_hops):
                idx = self.sequences[i, idx]
            self.targets[i] = idx
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target = self.targets[idx]
        
        # Text-encode: space-separated integers
        text = " ".join(str(x) for x in seq)
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(target, dtype=torch.long),  # Classification label
        }


def synthetic_dataset(task_name: str, tokenizer, training: bool = True, 
                      num_samples: int = 100000, seed: int = 42):
    """
    Load a synthetic dataset by name.
    
    Args:
        task_name: Either "polynomial" or "index_sum"
        tokenizer: BERT tokenizer
        training: If True, use training seed; if False, use validation seed
        num_samples: Number of samples (default 100k for train, 10k for val)
        seed: Base random seed
        
    Returns:
        Dataset object
    """
    # Use different seeds for train/val to ensure no overlap
    actual_seed = seed if training else seed + 1000
    actual_samples = num_samples if training else num_samples // 10
    
    if task_name == "polynomial":
        # Polynomial uses raw floats, no tokenizer needed
        return PolynomialDataset(
            num_samples=actual_samples,
            seed=actual_seed,
        )
    elif task_name == "index_sum":
        return IndexSumDataset(
            tokenizer=tokenizer,
            num_samples=actual_samples,
            seq_len=16,  # 3 indices + 13 values
            seed=actual_seed,
            max_seq_len=64,
        )
    elif task_name == "multiindex":
        return MultiIndexDataset(
            tokenizer=tokenizer,
            num_samples=actual_samples,
            seq_len=16,  # 16 classes
            seed=actual_seed,
            max_seq_len=64,
        )
    else:
        raise ValueError(f"Unknown synthetic task: {task_name}. Choose 'polynomial', 'index_sum', or 'multiindex'")


# Constants for integration with main.py
SYNTHETIC_TASKS = ["polynomial", "index_sum", "multiindex"]

# Tasks that use regression (vs classification)
SYNTHETIC_REGRESSION_TASKS = ["polynomial", "index_sum"]

# Tasks that use float inputs (bypass tokenization)
SYNTHETIC_FLOAT_INPUT_TASKS = ["polynomial"]


def is_synthetic_task(task_name: str) -> bool:
    """Check if a task name is a synthetic task."""
    return task_name in SYNTHETIC_TASKS


def is_synthetic_regression(task_name: str) -> bool:
    """Check if a synthetic task uses regression (vs classification)."""
    return task_name in SYNTHETIC_REGRESSION_TASKS


def uses_float_input(task_name: str) -> bool:
    """Check if a synthetic task uses float inputs (bypasses tokenization)."""
    return task_name in SYNTHETIC_FLOAT_INPUT_TASKS


def synthetic_num_classes(task_name: str) -> int:
    """Return number of classes for classification synthetic tasks."""
    if task_name == "multiindex":
        return 16  # seq_len
    return None  # Regression tasks


def synthetic_max_seq_length(task_name: str) -> int:
    """Return the max sequence length for synthetic tasks."""
    if task_name == "polynomial":
        return 16  # Sequence length for float input (needs >1 for BERT to work)
    elif task_name == "index_sum":
        return 64
    elif task_name == "multiindex":
        return 64
    return 64


def polynomial_collate_fn(batch):
    """Custom collate function for polynomial dataset (float inputs)."""
    return {
        "input_floats": torch.stack([item["input_floats"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }
