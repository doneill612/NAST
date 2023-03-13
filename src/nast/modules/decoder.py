import torch
import torch.nn as nn
import torch.nn.functional as functional

from typing import Union, Tuple, Optional
from torch import FloatTensor

from .attention import MultiheadAttention, positional_encoding_table
from .mlp import FeedForwardBlock
from ..config import NastTransformerConfig


class Decoder(nn.Module):
    
    def __init__(self, config: NastTransformerConfig):
        super(Decoder, self).__init__()
        self.config = config
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.encoder_blocks)])
    
    def forward(
        self, 
        hidden_states: FloatTensor, 
        key_value_states: Tuple[FloatTensor, FloatTensor], 
        attention_mask: Optional[FloatTensor]=None, 
        return_attention: bool=False
    ):
        dec_attentions = []
        for decoder_block in self.blocks:
            hidden_states, attn_spt, cross_attn = decoder_block(
                hidden_states=hidden_states,
                key_value_states=key_value_states,
                attention_mask=attention_mask,
                return_attention=True
            )
            if return_attention:
                dec_attentions.append((attn_spt, cross_attn))
        if return_attention:
            return hidden_states, dec_attentions[-1]
        return hidden_states

class DecoderBlock(nn.Module):
    
    def __init__(self, config: NastTransformerConfig):
        super(DecoderBlock, self).__init__()
        self.config = config

        self.pos_encode = nn.Embedding.from_pretrained(
            positional_encoding_table(config.prediction_length, config.embed_dim),
            freeze=True
        )
        self.self_attention = MultiheadAttention(
            num_heads=config.decoder_attn_heads,
            embed_dim=config.embed_dim,
            attn_dropout=config.decoder_attn_dropout,
            ff_dropout=config.decoder_ff_dropout
        )
        self.cross_attention = MultiheadAttention(
            num_heads=config.decoder_attn_heads,
            embed_dim=config.embed_dim,
            attn_dropout=config.decoder_attn_dropout,
            ff_dropout=config.decoder_ff_dropout
        )
        self.mlp = FeedForwardBlock(config.embed_dim, config.decoder_ff_expansion, ff_dropout=config.decoder_ff_dropout)

    def forward(
        self,
        hidden_states: FloatTensor,
        key_value_states: FloatTensor,
        attention_mask: Optional[FloatTensor]=None,
        return_attention: bool=False
    ):
        # create positional encoded to be used for temporal attention mechanism
        positions = torch.arange(0, self.config.prediction_length, dtype=torch.long).to(hidden_states.device)
        positional_encoding = self.pos_encode(positions)
        
        # add positional encoding to the hidden states for temporal attention 
        # reshape hidden states + pos encoding for consumption by attention module
        # no positional encoding required for spatial attention
        
        pos_encoded_hidden_states = (hidden_states + positional_encoding).transpose(1, 2).contiguous()

        if key_value_states is not None:
            key_value_states = key_value_states.transpose(1, 2).contiguous()

        # calculate temporal and spacial attentions
        decout, tattn = self.self_attention(
            pos_encoded_hidden_states, 
            key_value_states=None,
            attention_mask=attention_mask,
            axis='time'
        )
        _, sattn = self.self_attention(
            hidden_states,
            key_value_states=None,
            attention_mask=attention_mask, 
            axis='space'
        )

        # create "temporal influence map" from temporal attention weights
        # spacial attention weights @ temporal influence map => spatial-temporal attention
        attn_st = torch.matmul(sattn, tattn)
        # assert False, (attn_st.shape, decout.shape)
        decout = torch.matmul(
            attn_st,
            decout.transpose(1, 2).contiguous()
        )

        decout, cross_attn = self.cross_attention(
            decout.transpose(1, 2).contiguous(),
            key_value_states=key_value_states,
            attention_mask=attention_mask,
            axis='time'
        )

        decout = decout.transpose(1, 2).contiguous()

        # output shape (batch_size, nobj, prediction_length, embed_dim)
        decout = self.mlp(decout)

        if return_attention:
            return decout, attn_st, cross_attn
        return decout

