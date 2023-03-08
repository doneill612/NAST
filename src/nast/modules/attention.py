import numpy as np
import torch
import torch.nn as nn

from torch import FloatTensor
from torch.nn import functional

from typing import Optional, Union, Tuple 


def split_attn_heads(
    batch_size: int, 
    sequence_length:int, 
    num_heads: int,
    head_dim: int,
    *tensors: Union[FloatTensor, Tuple[FloatTensor]]
) -> Union[FloatTensor, Tuple[FloatTensor]]:
    """Reshapes supplied tensors (typically query, key, and value state tensors) for consumption 
    by a `ScaledDotProductAttention` module.

    The input tensors are assumed to have shape (batch_size, sequence_length, embed_dim).

    Args:
        `batch_size` (`int`)
            Batch size, dim=0 for the input tensors.
        `sequence_length` (`int`)
            Length of input sequences, dim=1 for tensors
        `num_heads` (`int`)
            The number of attention heads. Typically an instance parameter
            of the `MultiHeadAttention` module.
        `head_dim` (`int`)
            Equivalent to `embed_dim // num_heads`, also a property of 
            a `MultiheadAttention` module.
    Returns:
        a new set of reshaped tensors with shape (batch_size, num_heads, sequence_length, head_dim)
    """
    if isinstance(tensors, FloatTensor):
        tensors = (tensors,)
    return (
        tensor.view(
            batch_size,
            sequence_length,
            num_heads,
            head_dim
        ).transpose(1, 2).contiguous()
        for tensor in tensors
    )

def positional_encoding_table(sequence_length: int, embed_dim: int) -> FloatTensor:    
    positions = list(range(sequence_length))
    def cal_angle(position, hid_idx):
        return position / np.power(1000.0, 2 * (hid_idx // 2) / embed_dim)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(embed_dim)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table)
    
class ScaledDotProductAttention(nn.Module):

    def __init__(self, embed_dim: int, dropout: float=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = embed_dim ** -0.5
        self.dropout = dropout

    def forward(
        self, 
        queries: FloatTensor, 
        keys: FloatTensor, 
        values: FloatTensor, 
        mask: Optional[FloatTensor]=None
    ) -> FloatTensor:
        
        attn_weight = torch.matmul(queries * self.scale, keys.transpose(2, 3))
        if mask is not None:
            attn_weight = attn_weight.masked_fill(mask == 0, -1e9)

        attn = functional.softmax(attn_weight, dim=-1)
        attn = functional.dropout(attn, p=self.dropout, training=self.training)
        
        out = torch.matmul(attn, values)
        return out, attn
    
class MultiheadAttention(nn.Module):

    def __init__(
        self, 
        num_heads, 
        embed_dim, 
        attn_dropout=0.0, 
        embed_dropout=0.0,
        bias=True
    ):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.attn_dropout = attn_dropout
        self.embed_dropout = embed_dropout
        
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_proj = ScaledDotProductAttention(embed_dim, dropout=attn_dropout)

        self.fc = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        hidden_states: FloatTensor,
        key_value_states: Optional[Tuple[FloatTensor]]=None,
        attention_mask: Optional[FloatTensor]=None
    ):
        batch_size, seq_len = hidden_states.size(0), hidden_states.size(1)
        is_cross_attention = key_value_states is not None

        residual = hidden_states

        queries = self.query_proj(hidden_states)
        if is_cross_attention:
            keys = self.key_proj(key_value_states[0])
            values = self.value_proj(key_value_states[1])
        else:
            keys = self.key_proj(hidden_states)
            values = self.value_proj(hidden_states)
        
        queries, keys, values = split_attn_heads(
            batch_size, 
            seq_len, 
            self.num_heads, 
            self.head_dim, 
            queries, 
            keys, 
            values
        )

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)

        out, attn = self.attn_proj(queries, keys, values, mask=attention_mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        out = self.fc(out)
        out = functional.dropout(out, p=self.embed_dropout, training=self.training)
        out += residual
        out = self.layer_norm(out)

        return out, attn
