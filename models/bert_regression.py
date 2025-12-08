"""
BERT model with a regression head for synthetic tasks.

Wraps a pretrained BERT model and replaces the classification head with
a regression head that outputs a single float value.

Supports two input modes:
1. Token-based: standard input_ids (for text inputs)
2. Float-based: raw float values projected to embeddings (for polynomial task)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class BertForRegression(nn.Module):
    """
    BERT model with a regression head.
    
    Uses the [CLS] token representation to predict a single float value.
    Compatible with the pruning infrastructure (head_mask, ffn_mask).
    
    Supports direct float inputs via `input_floats` parameter, which bypasses
    tokenization and projects floats directly to BERT's hidden dimension.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", output_dim: int = 1, 
                 use_float_input: bool = False, float_seq_len: int = 1):
        """
        Args:
            model_name: Pretrained BERT model name
            output_dim: Output dimension (default 1 for scalar regression)
            use_float_input: If True, expect raw floats instead of token IDs
            float_seq_len: Sequence length when using float inputs
        """
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.use_float_input = use_float_input
        self.float_seq_len = float_seq_len
        
        # Float-to-embedding projection (for polynomial task)
        if use_float_input:
            # Project single float to hidden_size, then tile to seq_len
            self.float_projection = nn.Linear(1, self.config.hidden_size)
            self.float_projection.weight.data.normal_(mean=0.0, std=0.02)
            self.float_projection.bias.data.zero_()
            
            # Learnable position embeddings for float input
            self.float_position_embeddings = nn.Embedding(
                float_seq_len, self.config.hidden_size
            )
            self.float_layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
            self.float_dropout = nn.Dropout(self.config.hidden_dropout_prob)
        
        # Regression head: hidden_size -> output_dim
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.regressor = nn.Linear(self.config.hidden_size, output_dim)
        
        # Initialize regressor
        self.regressor.weight.data.normal_(mean=0.0, std=0.02)
        self.regressor.bias.data.zero_()
        
        # For compatibility with pruning code
        self.trainable_head_mask = None
        self.trainable_ffn_mask = None
        self.use_gumbel = False
        self.gumbel_temperature = 1.0
        
        # Store base_model_prefix for compatibility with arch.py
        self.base_model_prefix = "bert"
    
    def _create_float_embeddings(self, input_floats):
        """
        Convert raw float values to embeddings for BERT.
        
        Args:
            input_floats: (batch_size, 1) raw float values
            
        Returns:
            inputs_embeds: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        """
        batch_size = input_floats.shape[0]
        device = input_floats.device
        
        # Project float to hidden dimension: (batch, 1) -> (batch, hidden_size)
        float_embed = self.float_projection(input_floats)  # (batch, hidden_size)
        
        # Tile to sequence length: (batch, hidden_size) -> (batch, seq_len, hidden_size)
        float_embed = float_embed.unsqueeze(1).expand(-1, self.float_seq_len, -1)
        
        # Add position embeddings
        position_ids = torch.arange(self.float_seq_len, device=device).unsqueeze(0)
        position_embeds = self.float_position_embeddings(position_ids)
        
        # Combine and normalize
        embeddings = float_embed + position_embeds
        embeddings = self.float_layernorm(embeddings)
        embeddings = self.float_dropout(embeddings)
        
        # Create attention mask (all ones since all positions are valid)
        attention_mask = torch.ones(batch_size, self.float_seq_len, device=device)
        
        return embeddings, attention_mask
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        head_mask=None,
        labels=None,
        input_floats=None,
        **kwargs
    ):
        """
        Forward pass.
        
        Args:
            input_ids: (batch_size, seq_len) token IDs (for text input)
            attention_mask: (batch_size, seq_len) attention mask
            token_type_ids: (batch_size, seq_len) token type IDs
            head_mask: (num_layers, num_heads) head mask for pruning
            labels: (batch_size, 1) regression targets
            input_floats: (batch_size, 1) raw float values (for polynomial task)
            
        Returns:
            Object with .loss and .logits attributes
        """
        # Use trainable masks if available and no explicit mask provided
        if head_mask is None and self.trainable_head_mask is not None:
            if self.use_gumbel:
                from utils.schedule import gumbel_sigmoid
                head_mask = gumbel_sigmoid(
                    self.trainable_head_mask,
                    self.gumbel_temperature,
                    training=self.training
                )
            else:
                head_mask = torch.sigmoid(self.trainable_head_mask)
        
        # Handle float inputs (polynomial task)
        if input_floats is not None:
            inputs_embeds, attention_mask = self._create_float_embeddings(input_floats)
            outputs = self.bert(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                head_mask=head_mask,
            )
        else:
            # Standard token-based input
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                head_mask=head_mask,
            )
        
        # Get [CLS] token representation
        pooled_output = outputs.pooler_output  # (batch_size, hidden_size)
        pooled_output = self.dropout(pooled_output)
        
        # Regression prediction
        logits = self.regressor(pooled_output)  # (batch_size, output_dim)
        
        # Compute MSE loss if labels provided
        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(logits, labels)
        
        # Return object with loss and logits
        return RegressionOutput(loss=loss, logits=logits)


class RegressionOutput:
    """Simple output container for compatibility with HuggingFace-style outputs."""
    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits


def load_bert_for_regression(model_name: str = "bert-base-uncased", 
                              use_float_input: bool = False,
                              float_seq_len: int = 1):
    """
    Load a BERT model configured for regression.
    
    Args:
        model_name: Pretrained model name
        use_float_input: If True, model expects raw floats instead of tokens
        float_seq_len: Sequence length when using float inputs
        
    Returns:
        (config, model, tokenizer) tuple
        tokenizer is None if use_float_input=True
    """
    from transformers import AutoTokenizer
    
    model = BertForRegression(
        model_name=model_name,
        use_float_input=use_float_input,
        float_seq_len=float_seq_len,
    )
    
    if use_float_input:
        tokenizer = None  # No tokenizer needed for float inputs
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    config = model.config
    
    return config, model, tokenizer
