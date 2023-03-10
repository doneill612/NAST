import numpy as np
import torch
import torch.nn as nn

from torch import FloatTensor
from torch.nn import functional

from typing import Optional, Union, Tuple


def positional_encoding_table(sequence_length: Union[int, FloatTensor], embed_dim: int) -> FloatTensor:
    """Traditional sinusoidal positional encoding table proposed in Vaswani et al."""
    if isinstance(sequence_length, int):  
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
    """Scaled dot product attention mechanism.
    
    Follows similar implementation to Vaswani et al. Temperature scaling
    equal to `1 / sqrt(embed_dim)`.
    """
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
        
        attn = torch.matmul(queries * self.scale, keys.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = functional.softmax(attn, dim=-1)
        attn = functional.dropout(attn, p=self.dropout, training=self.training)
        
        out = torch.matmul(attn, values)
        return out, attn
    
class MultiheadAttention(nn.Module):
    """Multiheaded attention module adapted to perform both spatial and temporal 
    scaled dot product attention with multiple attention heads. The attention weights are 
    mean-scaled along the head dimension after computation.

    There are small additions to the forward pass which reshape the input hidden states 
    accordingly depending on the attention 'axis,' which can be either 'time' or 'space.'
    
    In spatial attention, time steps are consumed into the batch dimension and the attention
    mechanism is applied across features in the input sequences. In temporal attention (similar 
    to the canoncial transformer architecture), features are consumed into the batch dimension and
    the attention mechanism is applied across each timestep.
    """
    def __init__(
        self, 
        num_heads, 
        embed_dim, 
        attn_dropout=0.0, 
        ff_dropout=0.0,
        bias=True
    ):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_proj = ScaledDotProductAttention(embed_dim, dropout=attn_dropout)

        self.fc = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        hidden_states: FloatTensor,
        key_value_states: Optional[Tuple[FloatTensor, FloatTensor]]=None,
        attention_mask: Optional[FloatTensor]=None,
        axis: str='time'
    ):
        batch_size, seq_len, channels, embed_dim = hidden_states.size()
        is_cross_attention = key_value_states is not None
        residual = hidden_states

        if axis == 'space':
            # sptial attention, collapse timesteps into batch dimension and attend 
            # to each timeseries feature (channel)
            hidden_states = hidden_states.view(-1, channels, embed_dim)
        else:
            # temporal attention, transpose channel and time dimensions,
            # collapse channels into batch dimension and attend to 
            # each individual timestep
            hidden_states = hidden_states.transpose(1, 2).contiguous().view(-1, seq_len, embed_dim)

        queries: FloatTensor = self.query_proj(hidden_states)
        if is_cross_attention:
            ks, vs = key_value_states
            if axis == 'space':
                # spatial attention, same as above, attend to channels
                ks = ks.view(-1, channels, embed_dim)
                vs = vs.view(-1, channels, embed_dim)
            else:
                # temporal attention, same as above, attend to timesteps
                ks = ks.transpose(1, 2).contiguous().view(-1, seq_len, embed_dim)
                vs = vs.transpose(1, 2).contiguous().view(-1, seq_len, embed_dim)
            keys: FloatTensor = self.key_proj(ks)
            values: FloatTensor = self.value_proj(vs)
        else:
            keys: FloatTensor = self.key_proj(hidden_states)
            values: FloatTensor = self.value_proj(hidden_states)
        
        # split to each attention head
        if axis == 'time':
            queries = queries.view(-1, self.num_heads, seq_len, self.head_dim)
            keys = keys.view(-1, self.num_heads, seq_len, self.head_dim)
            values = values.view(-1, self.num_heads, seq_len, self.head_dim)
        else:
            queries = queries.view(-1, self.num_heads, channels, self.head_dim)
            keys = keys.view(-1, self.num_heads, channels, self.head_dim)
            values = values.view(-1, self.num_heads, channels, self.head_dim)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)

        out, attn = self.attn_proj(queries, keys, values, mask=attention_mask)
        
        # reshape, average attention across heads
        out = out.view(batch_size, seq_len, channels, embed_dim)
        if axis == 'space':
            attn = attn.view(batch_size, seq_len, self.num_heads, channels, channels)
        else:
            attn = attn.view(batch_size, channels, self.num_heads, seq_len, seq_len)
        attn = torch.mean(attn, dim=2)
        
        # fc + residual
        out = self.fc(out)
        out = functional.dropout(out, p=self.ff_dropout, training=self.training)
        out += residual

        # layer norm
        out = self.layer_norm(out)

        return out, attn
