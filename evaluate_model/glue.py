import torch
from evaluate import load as load_metric
from utils.arch import apply_neuron_mask
from dataset.glue import target_dev_metric


@torch.no_grad()
def eval_glue_acc(model, head_mask, ffn_intermediate_mask, ffn_output_mask, dataloader, task_name):
    IS_STSB = model.num_labels == 1
    metric = load_metric("glue", task_name)

    model.eval()
    handles_int = apply_neuron_mask(model, ffn_intermediate_mask, type="ffn_1")
    handles_out = apply_neuron_mask(model, ffn_output_mask, type="ffn_2")
    # handles_attn = apply_neuron_mask(model, head_mask, type="attn")
    for batch in dataloader:
        for k, v in batch.items():
            batch[k] = v.to("cuda", non_blocking=True)

        # print(head_mask)
        outputs = model(head_mask=head_mask, **batch)
        # print("logits", outputs.logits)
        if IS_STSB:
            predictions = outputs.logits.squeeze()
        else:
            predictions = outputs.logits.argmax(dim=-1)
        metric.add_batch(
            predictions=predictions,
            references=batch["labels"],
        )
    for handle in handles_int:
        handle.remove()
    for handle in handles_out:
        handle.remove()
    # for handle in handles_attn:
    #     handle.remove()

    eval_results = metric.compute()
    target_metric = target_dev_metric(task_name)
    accuracy = eval_results[target_metric]
    return accuracy
