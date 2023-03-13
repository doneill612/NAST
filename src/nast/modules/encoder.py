import torch
import torch.nn as nn
import torch.nn.functional as functional

from typing import Union, Tuple, Optional
from torch import FloatTensor

from .attention import MultiheadAttention, positional_encoding_table
from .mlp import FeedForwardBlock
from ..config import NastTransformerConfig

class Encoder(nn.Module):

    def __init__(self, config: NastTransformerConfig):
        super().__init__()    
        self.config = config
        self.channel_embed = nn.Linear(config.channels, config.embed_dim)
        self.layernorm = nn.LayerNorm(config.embed_dim)
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.encoder_blocks)])

    def forward(
        self, 
        input_sequences: FloatTensor, 
        return_attention: bool=True
    ):
        hidden_states = self.channel_embed(input_sequences)
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
        
class EncoderBlock(nn.Module):

    def __init__(self, config: NastTransformerConfig):
        super(EncoderBlock, self).__init__()
        
        self.config = config
    
        self.pos_encode = nn.Embedding.from_pretrained(
            positional_encoding_table(config.context_length, config.embed_dim),
            freeze=True
        )
        self.attention = MultiheadAttention(
            num_heads=config.encoder_attn_heads,
            embed_dim=config.embed_dim,
            attn_dropout=config.encoder_attn_dropout,
            ff_dropout=config.encoder_ff_dropout
        )
        self.mlp = FeedForwardBlock(config.embed_dim, config.encoder_ff_expansion, ff_dropout=config.encoder_ff_dropout)

    def forward(
        self, 
        hidden_states: FloatTensor,
        key_value_states: Optional[FloatTensor]=None,
        attention_mask: Optional[FloatTensor]=None,
        return_attention: bool=False,
    ) -> Union[FloatTensor, Tuple[FloatTensor, FloatTensor]]:
        # create positional encoded to be used for temporal attention mechanism
        positions = torch.arange(0, self.config.context_length, dtype=torch.long).to(hidden_states.device)
        positional_encoding = self.pos_encode(positions)
        
        # add positional encoding to the hidden states for temporal attention 
        # reshape hidden states + pos encoding for consumption by attention module
        pos_encoded_hidden_states = (hidden_states + positional_encoding).transpose(1, 2).contiguous()
        # no positional encoding required for spatial attention
        # reshape hidden states for consumption by attention module
        hidden_states = hidden_states.transpose(1, 2).contiguous()

        if key_value_states:
            key_value_states = key_value_states.transpose(1, 2).contiguous()

        # calculate temporal and spacial attentions
        encout, tattn = self.attention(
            pos_encoded_hidden_states, 
            key_value_states=key_value_states,
            attention_mask=attention_mask,
            axis='time'
        )
        _, sattn = self.attention(
            hidden_states,
            key_value_states=key_value_states,
            attention_mask=attention_mask, 
            axis='space'
        )

        # create "temporal influence map" from temporal attention weights
        # spacial attention weights @ temporal influence map => spatial-temporal attention
        attn_st = torch.matmul(sattn, tattn.transpose(1, 2).contiguous())
        encout = torch.matmul(
            attn_st.transpose(1, 2).contiguous(),
            encout.transpose(1, 2).contiguous()
        )

        # output shape (batch_size, nobj, context_length, embed_dim)
        encout = self.mlp(encout)

        if return_attention:
            return encout, attn_st
        
        return encout


