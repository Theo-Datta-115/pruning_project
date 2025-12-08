import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from dataset.glue import glue_dataloader
from dataset.squad import squad_test_dataloader
from dataset.synthetic import (
    synthetic_dataset, is_synthetic_task, is_synthetic_regression,
    uses_float_input, polynomial_collate_fn
)
from evaluate_model.glue import eval_glue_acc
from evaluate_model.squad import eval_squad_acc


@torch.no_grad()
def eval_synthetic_mse(model, head_mask, ffn_mask, dataloader, use_float_input=False):
    """
    Evaluate synthetic regression task using MSE loss.
    
    Returns negative MSE (so higher is better, consistent with accuracy metrics).
    """
    model.eval()
    total_mse = 0.0
    total_samples = 0
    
    for batch in dataloader:
        labels = batch["labels"].cuda()
        
        if use_float_input:
            # Float input mode (polynomial task)
            input_floats = batch["input_floats"].cuda()
            outputs = model(
                input_floats=input_floats,
                labels=labels,
                head_mask=head_mask,
            )
        else:
            # Token input mode
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                head_mask=head_mask,
            )
        
        batch_size = labels.shape[0]
        total_mse += outputs.loss.item() * batch_size
        total_samples += batch_size
    
    avg_mse = total_mse / total_samples
    # Return negative MSE so that higher is better (consistent with accuracy)
    return -avg_mse


@torch.no_grad()
def eval_synthetic_accuracy(model, head_mask, ffn_mask, dataloader):
    """
    Evaluate synthetic classification task using accuracy.
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    
    for batch in dataloader:
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        labels = batch["labels"].cuda()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        
        predictions = outputs.logits.argmax(dim=-1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.shape[0]
    
    return total_correct / total_samples


@torch.no_grad()
def test_accuracy(model, head_mask, ffn_mask, tokenizer, task_name):
    IS_SQUAD = "squad" in task_name
    IS_SYNTHETIC = is_synthetic_task(task_name)

    test_batch_size = 32 if IS_SQUAD else 128
    
    if IS_SYNTHETIC:
        # Create validation dataset for synthetic tasks
        val_dataset = synthetic_dataset(
            task_name,
            tokenizer,
            training=False,
            num_samples=100000,  # Will use 10k for validation
            seed=42,
        )
        
        # Use appropriate collate function based on input type
        use_floats = uses_float_input(task_name)
        if use_floats:
            collate_fn = polynomial_collate_fn
        else:
            collate_fn = DataCollatorWithPadding(tokenizer)
        
        test_dataloader = DataLoader(
            val_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        
        if is_synthetic_regression(task_name):
            return eval_synthetic_mse(model, head_mask, ffn_mask, test_dataloader, use_float_input=use_floats)
        else:
            return eval_synthetic_accuracy(model, head_mask, ffn_mask, test_dataloader)
    elif IS_SQUAD:
        eval_dataset, eval_examples, test_dataloader = squad_test_dataloader(
            task_name,
            tokenizer,
            batch_size=test_batch_size,
            pad_to_max=False,
        )
        acc = eval_squad_acc(
            model,
            head_mask,
            ffn_mask,
            test_dataloader,
            eval_dataset,
            eval_examples,
            task_name,
        )
    else:
        test_dataloader = glue_dataloader(
            task_name,
            tokenizer,
            training=False, 
            batch_size=test_batch_size,
            pad_to_max=False,
        )
        acc = eval_glue_acc(
            model,
            head_mask,
            ffn_mask,
            test_dataloader,
            task_name,
        )
    return acc
