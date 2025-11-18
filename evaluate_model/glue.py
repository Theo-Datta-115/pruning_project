import torch
from evaluate import load as load_metric
from utils.arch import apply_neuron_mask
from dataset.glue import target_dev_metric


@torch.no_grad()
def eval_glue_acc(model, head_mask, ffn_mask, dataloader, task_name):
    IS_STSB = model.num_labels == 1
    metric = load_metric("glue", task_name)

    model.eval()
    handles_ffn = apply_neuron_mask(model, ffn_mask, type="ffn_2")
    for batch in dataloader:
        for k, v in batch.items():
            batch[k] = v.to("cuda", non_blocking=True)

        outputs = model(head_mask=head_mask, **batch)
        if IS_STSB:
            predictions = outputs.logits.squeeze()
        else:
            predictions = outputs.logits.argmax(dim=-1)
        metric.add_batch(
            predictions=predictions,
            references=batch["labels"],
        )
    for handle in handles_ffn:
        handle.remove()

    eval_results = metric.compute()
    target_metric = target_dev_metric(task_name)
    accuracy = eval_results[target_metric]
    return accuracy
