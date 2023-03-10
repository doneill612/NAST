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
    scaled dot product attention with multiple attention heads.
     
    When this module is used as part of a spatial-temporal self-attention encoder or 
    decoder block, the attention mechanism is called twice for both spatial and 
    temporal attention, and the two sets of weights are used to create a temporal influence 
    map which is applied to the encoder outputs.
    
    The attention weights are mean-scaled along the head dimension after computation.
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
        pe_hidden_states: Optional[FloatTensor]=None,
        key_value_states: Optional[Tuple[FloatTensor, FloatTensor]]=None,
        attention_mask: Optional[FloatTensor]=None,
    ):
        """Forward pass.
        
        Args
        ----
            `hidden_states` (`FloatTensor`)
                Input hidden states. If `pe_hidden_states` not supplied,
                assumed to be a tensor with added positional encoding
            `pe_hidden_states` (`FloatTensor`, optional)
                Positionally encoded hidden states. If supplied, 
                these are treated as the positionally encoded hidden states
                to be used for temporal attention, and the `hidden_states`
                arg is considered *not* to be positionally encoded and used
                for spatial attention
            `key_value_states` (`Tuple[FloatTensor, FloatTensor]`, optional)
                If supplied, assumed to be encoder key and value states used in 
                decoder for cross attention, with `key_value_states[0]` being 
                the key states and `key_value_states[1]` being the value states
            `attention_mask` (`FloatTensor`, optional)
                Optional attention mask to apply in scaled dot product attention
                mehcanism
        
        Returns
        -------
            The attention outputs and either canoncial temporal attention weights or 
            spatial-temporal weights (if `pe_hidden_states` was supplied)
        """
        is_spatial_temporal = pe_hidden_states is not None
        residual = pe_hidden_states if is_spatial_temporal else hidden_states
        if is_spatial_temporal and tuple(hidden_states.shape) != tuple(pe_hidden_states.shape):
            raise ValueError(
                f'Shape mismatch between temporal and spatial hidden states: '
                f'{hidden_states.shape} != {pe_hidden_states.shape}'
            )
        batch_size, seq_len, embed_dim = hidden_states.size()
        is_cross_attention = key_value_states is not None

        def apply_attention(
            tensor: FloatTensor, 
            attention_mask: Optional[FloatTensor]=None, 
        ):
            queries: FloatTensor = self.query_proj(tensor)
            if is_cross_attention:
                ks, vs = key_value_states
                keys: FloatTensor = self.key_proj(ks)
                values: FloatTensor = self.value_proj(vs)
            else:
                keys: FloatTensor = self.key_proj(tensor)
                values: FloatTensor = self.value_proj(tensor)
            
            # split to each attention head
            queries = queries.view(-1, seq_len, self.num_heads, self.head_dim)
            keys = keys.view(-1, seq_len, self.num_heads, self.head_dim)
            values = values.view(-1, seq_len, self.num_heads, self.head_dim)
            

            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1)

            out, attn = self.attn_proj(
                queries.transpose(1, 2).contiguous(), 
                keys.transpose(1, 2).contiguous(), 
                values.transpose(1, 2).contiguous(), 
                mask=attention_mask
            )
            
            # reshape, average attention across heads
            out = out.view(batch_size, seq_len, embed_dim)
            # attn = attn.view(batch_size, seq_len, self.num_heads, embed_dim)
            attn = torch.mean(attn, dim=1)

            return out, attn

        if is_spatial_temporal:
            tout, tattn = apply_attention(pe_hidden_states, attention_mask)
            _, sattn = apply_attention(hidden_states, attention_mask)
            # create "temporal influence map" from temporal attention weights
            # spacial attention weights @ temporal influence map => spatial-temporal attention
            attn_st = torch.matmul(sattn, tattn)
            out = torch.matmul(attn_st.mT, tout)
            out = self.fc(out)
            out = functional.dropout(out, p=self.ff_dropout, training=self.training)
            out += residual
            out = self.layer_norm(out)
            return out, attn_st
        else:
            # Only doing canoncial temporal attention
            out, attn = apply_attention(hidden_states, attention_mask)
            out = self.fc(out)
            out = functional.dropout(out, p=self.ff_dropout, training=self.training)
            out += residual
            out = self.layer_norm(out)
            return out, attn
        