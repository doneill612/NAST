import torch
import torch.nn as nn

from typing import Tuple, Optional
from torch import FloatTensor

from .attention import MultiheadAttention, positional_encoding_table
from ..config import NastTransformerConfig

class SptialTemporalEncoder(nn.Module):

    def __init__(self, config: NastTransformerConfig):
        super().__init__()    
        self.config = config
        self.blocks = nn.ModuleList([SpatialTemporalEncoderBlock(config) for _ in range(config.encoder_blocks)])

    def forward(self, input_sequences: FloatTensor):
        hidden_states = input_sequences.unsqueeze(1).repeat_interleave(self.config.embed_dim, dim=-1)
        hidden_states = hidden_states.transpose(1, 2).contiguous()
        # TODO finish, just a half-baked method at the moment
        for idx, encoder_block in enumerate(self.blocks):
            out = encoder_block(
                hidden_states=hidden_states,
                return_attentions=True
            )

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
        self.expand = nn.Linear()

    def forward(
        self, 
        hidden_states: FloatTensor,
        return_attentions: bool=False
    ) -> Tuple[FloatTensor]:        
        hidden_states = self.input_layernorm(hidden_states)        
        
        positions = torch.arange(0, self.config.context_length, dtype=torch.long).to(hidden_states.device)
        positional_encoding = self.pos_encode(positions)
        
        temporal_attention_sequences = (hidden_states + positional_encoding).transpose(1, 2).contiguous()
        spatial_attention_sequences = hidden_states.transpose(1, 2).contiguous()

        temporal_attention_sequences, temporal_attention = self.attention(temporal_attention_sequences, axis='time')
        spatial_attention_sequences, spatial_attention = self.attention(spatial_attention_sequences, axis='space')

        out = [temporal_attention_sequences, spatial_attention_sequences,]
        if return_attentions:
            out.extend((temporal_attention, spatial_attention))
        return out


