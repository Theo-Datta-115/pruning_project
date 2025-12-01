import torch


def logit(x):
    """Inverse of sigmoid function (log-odds).
    
    Args:
        x: Tensor with values in (0, 1)
        
    Returns:
        logit(x) = log(x / (1 - x))
    """
    return torch.log(x / (1 - x))


@torch.no_grad()
def remove_padding(hidden_states, attention_mask):
    attention_mask = attention_mask.view(-1)
    nonzero = torch.nonzero(attention_mask).squeeze()
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[2])
    hidden_states = torch.index_select(hidden_states, dim=0, index=nonzero).contiguous()
    return hidden_states


def get_backbone(model):
    model_type = model.base_model_prefix
    backbone = getattr(model, model_type)
    return backbone


def get_encoder(model):
    backbone = get_backbone(model)
    encoder = backbone.encoder
    return encoder


def get_layers(model):
    encoder = get_encoder(model)
    layers = encoder.layer
    return layers

def get_attn(model, index):
    layer = get_layers(model)[index]
    attn = layer.attention
    return attn.self

def get_mha_proj(model, index):
    layer = get_layers(model)[index]
    mha_proj = layer.attention.output
    return mha_proj


def get_ffn1(model, index):
    layer = get_layers(model)[index]
    ffn1 = layer.intermediate
    return ffn1

def get_ffn2(model, index):
    layer = get_layers(model)[index]
    ffn2 = layer.output
    return ffn2


def get_classifier(model):
    backbone = get_backbone(model)
    if backbone.pooler is not None:
        classifier = model.classifier
    else:
        classifier = model.classifier.out_proj
    return classifier


def register_mask(module, mask):
    def hook(_, inputs):
        hidden_states = inputs[0]
        mask_view = mask.reshape(1, 1, -1).to(hidden_states.device, hidden_states.dtype)
        masked_hidden = hidden_states * mask_view
        if len(inputs) == 1:
            return (masked_hidden,)
        return (masked_hidden,) + tuple(inputs[1:])

    # print("mask", mask)
    # print("module", module)
    handle = module.register_forward_pre_hook(hook)
    return handle

def apply_neuron_mask(model, neuron_mask, type="ffn_2"):
    # If no mask is provided, return empty handles (no masking)
    if neuron_mask is None:
        return []
    
    num_hidden_layers = neuron_mask.shape[0]
    handles = []
    for layer_idx in range(num_hidden_layers):
        if type == "ffn_1":
            ffn = get_ffn1(model, layer_idx)
        elif type == "ffn_2":
            ffn = get_ffn2(model, layer_idx)
        else:
            raise ValueError(f"Unknown FFN mask type: {type}")

        mask_row = neuron_mask[layer_idx].reshape(-1)
        handles.append(register_mask(ffn, mask_row))
    return handles

class MaskNeurons:
    def __init__(self, model, neuron_mask):
        self.handles = apply_neuron_mask(model, neuron_mask)

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        for handle in self.handles:
            handle.remove()


def hijack_input(module, list_to_append):
    hook = lambda _, inputs: list_to_append.append(inputs)
    handle = module.register_forward_pre_hook(hook)
    return handle


@torch.no_grad()
def collect_layer_inputs(
    model,
    head_mask,
    neuron_mask,
    layer_idx,
    prev_inputs,
):
    layers = get_layers(model)
    target_layer = layers[layer_idx]

    inputs = []
    if layer_idx == 0:
        encoder = get_encoder(model)
        layers = encoder.layer
        encoder.layers = layers[:1]

        handle = hijack_input(target_layer, inputs)
        for batch in prev_inputs:
            for k, v in batch.items():
                batch[k] = v.to("cuda")
            with MaskNeurons(model, neuron_mask):
                model(head_mask=head_mask, **batch)

        handle.remove()
        encoder.layers = layers
        inputs = [list(x) for x in inputs]
    else:
        prev_layer = layers[layer_idx - 1]

        for batch in prev_inputs:
            batch[2] = head_mask[layer_idx - 1].view(1, -1, 1, 1)
            with MaskNeurons(model, neuron_mask):
                prev_output = prev_layer(*batch)

            batch[0] = prev_output[0]
            batch[2] = head_mask[layer_idx].view(1, -1, 1, 1)
            inputs.append(batch)

    return inputs


def count_parameters(module):
    """Count total parameters in a module."""
    return sum(p.numel() for p in module.parameters())


def get_attention_param_ratio(model):
    """
    Calculate the ratio between FFN block parameters and attention block parameters in a transformer model.
    
    Args:
        model: A transformer model (e.g., BERT)
        
    Returns:
        float: Ratio of FFN parameters to attention parameters (FFN_params / attention_params)
    """
    total_ffn_params = 0
    total_attention_params = 0
    
    # Get all layers
    layers = get_layers(model)
    num_layers = len(layers)
    
    for layer_idx in range(num_layers):
        # FFN parameters (intermediate + output layers)
        ffn1 = get_ffn1(model, layer_idx)  # intermediate dense layer
        ffn2 = get_ffn2(model, layer_idx)  # output dense layer
        total_ffn_params += count_parameters(ffn1) + count_parameters(ffn2)
        
        # Attention parameters (query, key, value, and output projection)
        attn = get_attn(model, layer_idx)  # self-attention (contains query, key, value)
        mha_proj = get_mha_proj(model, layer_idx)  # attention output projection
        total_attention_params += count_parameters(attn) + count_parameters(mha_proj)
    
    # Calculate ratio
    if total_attention_params == 0:
        return float('inf')  # Avoid division by zero
    
    ratio = total_attention_params / total_ffn_params
    return ratio
