import torch
import torch.nn as nn

from torch import FloatTensor

from .attention import MultiheadAttention, positional_encoding_table
from ..config import NastTransformerConfig


class SpatialTemporalEncoderBlock(nn.Module):

    def __init__(self, config: NastTransformerConfig):
        
        self.config = config
    
        self.pos_encode = nn.Embedding.from_pretrained(
            positional_encoding_table(config.context_length, config.embed_dim),
            freeze=True
        )
        self.input_layernorm = nn.LayerNorm(config.context_length)
        self.input_embed = nn.Sequential(
            nn.Conv1d(config.channels, config.embed_dim, kernel_size=1),
            nn.LayerNorm((config.embed_dim, config.context_length))
        )
        self.attention = MultiheadAttention(
            num_heads=config.encoder_attn_heads,
            embed_dim=config.embed_dim,
            attn_dropout=config.decoder_attn_dropout,
            ff_dropout=config.encoder_ff_dropout
        )

    def forward(self, input_sequences: FloatTensor):
        batch_size, *_ = input_sequences.size()
        residual = input_sequences
        input_sequences = self.input_layernorm(input_sequences)
        positions = torch.arange(
            0, self.config.context_length, dtype=torch.long
        ).expand(batch_size, self.config.context_length).to(input_sequences.device)
        hidden_states = input_sequences + self.pos_encode(positions)
        hidden_states, temporal_attention = self.attention(hidden_states)

