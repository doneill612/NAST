import pytest
import torch
import numpy as np

from torch import FloatTensor
from torch import nn

from nast.modules.qgb import QueryGenerationBlock
from nast.modules.attention import (
    ScaledDotProductAttention, 
    MultiheadAttention,
    positional_encoding_table,
)
from nast.modules.encoder import SpatialTemporalEncoderBlock
from nast.config import NastTransformerConfig

@pytest.fixture
def config():
    return NastTransformerConfig(
        context_length=15,
        prediction_length=5,
        channels=3
    )

@pytest.fixture
def input_sequences(config, batch_size):
    ts = np.zeros(shape=(config.channels, config.context_length))
    ts = FloatTensor(config.channels, config.context_length)
    ts = FloatTensor(ts)
    return ts.expand(batch_size, *ts.size())

@pytest.fixture
def hidden_states(config, batch_size):
    return torch.zeros((batch_size, config.channels, config.context_length, config.embed_dim))

@pytest.fixture
def batch_size():
    return 12

def test_positional_encoding_table(config, input_sequences, batch_size):
    sequence_length = config.context_length
    embed_dim = config.embed_dim
    channels = config.channels
    
    enc_table = positional_encoding_table(sequence_length, embed_dim)
    
    assert enc_table.size(0) == sequence_length, enc_table.size()
    assert enc_table.size(1) == embed_dim, enc_table.size()

    pos_embedding = nn.Embedding.from_pretrained(enc_table, freeze=True)
    pos = torch.arange(0, sequence_length, dtype=torch.long)
    
    input_sequences = input_sequences.unsqueeze(-1).repeat_interleave(embed_dim, dim=-1)
    input_sequences = input_sequences + pos_embedding(pos)
    input_sequences = input_sequences.transpose(1, 2).contiguous()

    assert tuple(input_sequences.shape) == (batch_size, sequence_length, channels, embed_dim)

def test_encoder_block(config, hidden_states, batch_size):
    block = SpatialTemporalEncoderBlock(config)
    temp, spat = block(hidden_states=hidden_states)
    assert tuple(temp.shape) == (batch_size, config.context_length, config.channels, config.embed_dim)
    assert tuple(spat.shape) == (batch_size, config.context_length, config.channels, config.embed_dim)