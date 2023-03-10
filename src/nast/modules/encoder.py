import torch
import torch.nn as nn
import torch.nn.functional as functional

from typing import Union, Tuple, Optional
from torch import FloatTensor

from .attention import MultiheadAttention, positional_encoding_table
from .mlp import FeedForwardBlock
from ..config import NastTransformerConfig

class SpatialTemporalEncoder(nn.Module):

    def __init__(self, config: NastTransformerConfig):
        super().__init__()    
        self.config = config
        self.conv_embedding = nn.Conv1d(config.channels, config.embed_dim, kernel_size=1)
        self.layernorm = nn.LayerNorm(config.embed_dim)
        self.blocks = nn.ModuleList([SpatialTemporalEncoderBlock(config) for _ in range(config.encoder_blocks)])

    def forward(
        self, 
        input_sequences: FloatTensor, 
        return_attention: bool=True
    ):
        hidden_states = input_sequences.unsqueeze(-1).repeat_interleave(self.config.embed_dim, dim=-1)
        hidden_states = functional.dropout(hidden_states, p=self.config.encoder_ff_dropout, training=self.training)
        hidden_states = self.layernorm(hidden_states)
        enc_self_attentions = []
        for encoder_block in self.blocks:
            hidden_states, attn_spt = encoder_block(
                hidden_states=hidden_states,
                return_attention=True
            )
            if return_attention:
                enc_self_attentions.append(attn_spt)
        if return_attention:
            return hidden_states, enc_self_attentions[-1]
        return hidden_states
        
class SpatialTemporalEncoderBlock(nn.Module):

    def __init__(self, config: NastTransformerConfig):
        super(SpatialTemporalEncoderBlock, self).__init__()
        
        self.config = config
    
        self.pos_encode = nn.Embedding.from_pretrained(
            positional_encoding_table(config.context_length, config.embed_dim),
            freeze=True
        )
        self.input_layernorm = nn.LayerNorm(config.embed_dim)
        self.attention = MultiheadAttention(
            num_heads=config.encoder_attn_heads,
            embed_dim=config.embed_dim,
            attn_dropout=config.decoder_attn_dropout,
            ff_dropout=config.encoder_ff_dropout
        )
        self.mlp = FeedForwardBlock(config.embed_dim, config.encoder_ff_expansion, ff_dropout=config.encoder_ff_dropout)

    def forward(
        self, 
        hidden_states: FloatTensor,
        key_value_states: Optional[Tuple[FloatTensor, FloatTensor]]=None,
        return_attention: bool=False,
    ) -> Union[FloatTensor, Tuple[FloatTensor, FloatTensor]]:
        # layernorm
        hidden_states = self.input_layernorm(hidden_states)        
        
        # create positional encoded to be used for temporal attention mechanism
        positions = torch.arange(0, self.config.context_length, dtype=torch.long).to(hidden_states.device)
        positional_encoding = self.pos_encode(positions)
        
        # add positional encoding to the hidden states for temporal attention 
        # reshape hidden states + pos encoding for consumption by attention module
        pos_encoded_hidden_states = (hidden_states + positional_encoding).transpose(1, 2).contiguous()
        # no positional encoding required for spatial attention
        # reshape hidden states for consumption by attention module
        reshaped_hidden_states = hidden_states.transpose(1, 2).contiguous()

        # calculate temporal and spacial attentions
        encout, tattn = self.attention(
            pos_encoded_hidden_states, key_value_states=key_value_states, axis='time'
        )
        _, sattn = self.attention(
            reshaped_hidden_states, key_value_states=key_value_states, axis='space'
        )

        # create "temporal influence map" from temporal attention weights
        # spacial attention weights @ temporal influence map => spatial-temporal attention
        attn_st = torch.matmul(sattn, tattn.transpose(1, 2).contiguous())
        encout = torch.matmul(
            attn_st.transpose(1, 2).contiguous(),
            encout.transpose(1, 2).contiguous()
        )

        # output shape (batch_size, channels, context_length, embed_dim)
        encout = self.mlp(encout)

        if return_attention:
            return encout, attn_st
        
        return encout


