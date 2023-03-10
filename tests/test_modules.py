import pytest
import torch
import numpy as np

from torch import FloatTensor
from torch import nn

from nast.modules.qgb import QueryGenerationBlock
from nast.modules.attention import positional_encoding_table
from nast.modules.encoder import SpatialTemporalEncoderBlock, Encoder
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
    ts = np.zeros(shape=(config.context_length, config.channels))
    ts = FloatTensor(config.context_length, config.channels)
    ts = FloatTensor(ts)
    return ts.expand(batch_size, *ts.size())

@pytest.fixture
def hidden_states(config, batch_size):
    return torch.randn((batch_size, config.context_length, config.embed_dim))

@pytest.fixture
def batch_size():
    return 12

def test_positional_encoding_table(config):
    sequence_length = config.context_length
    embed_dim = config.embed_dim
    channels = config.channels
    
    enc_table = positional_encoding_table(sequence_length, embed_dim)
    
    assert enc_table.size(0) == sequence_length, enc_table.size()
    assert enc_table.size(1) == embed_dim, enc_table.size()

    pos_embedding = nn.Embedding.from_pretrained(enc_table, freeze=True)
    pos = torch.arange(0, sequence_length, dtype=torch.long)
    pos = pos_embedding(pos)
    assert tuple(pos.shape) == (sequence_length, embed_dim), pos.shape()


def test_encoder_block(config, hidden_states, batch_size):
    block = SpatialTemporalEncoderBlock(config)
    encout = block(hidden_states=hidden_states, return_attention=False)
    assert tuple(encout.shape) == tuple(hidden_states.shape)

def test_encoder(config, input_sequences, batch_size):
    encoder = Encoder(config)
    hidden_states = encoder(input_sequences, return_attention=False)
    assert tuple(hidden_states.shape) == (batch_size, config.context_length, config.embed_dim)
