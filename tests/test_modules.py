import pytest
import torch
import numpy as np

from torch import FloatTensor
from torch import nn

from nast.modules.qgb import QueryGenerationBlock
from nast.modules.attention import positional_encoding_table
from nast.modules.encoder import EncoderBlock, Encoder
from nast.modules.decoder import DecoderBlock, Decoder
from nast.config import NastTransformerConfig

@pytest.fixture
def config():
    return NastTransformerConfig(
        context_length=15,
        prediction_length=5,
        num_objects=3,
        channels=5,
    )

@pytest.fixture
def input_sequences(config, batch_size):
    ts = np.zeros(shape=(config.num_objects, config.context_length, config.channels))
    ts = FloatTensor(config.num_objects, config.context_length, config.channels)
    ts = FloatTensor(ts)
    return ts.expand(batch_size, *ts.size())

@pytest.fixture
def hidden_states(config, batch_size):
    return torch.randn((batch_size, config.num_objects, config.context_length, config.embed_dim))

@pytest.fixture
def batch_size():
    return 12

def test_positional_encoding_table(config, input_sequences, batch_size):
    sequence_length = config.context_length
    embed_dim = config.embed_dim
    num_objects = config.num_objects
    
    enc_table = positional_encoding_table(sequence_length, embed_dim)
    
    assert enc_table.size(0) == sequence_length, enc_table.size()
    assert enc_table.size(1) == embed_dim, enc_table.size()

    pos_embedding = nn.Embedding.from_pretrained(enc_table, freeze=True)
    pos = torch.arange(0, sequence_length, dtype=torch.long)
    
    channel_embedding = nn.Linear(config.channels, config.embed_dim)
    input_sequences = channel_embedding(input_sequences)
    input_sequences = input_sequences + pos_embedding(pos)
    input_sequences = input_sequences.transpose(1, 2).contiguous()

    assert tuple(input_sequences.shape) == (batch_size, sequence_length, num_objects, embed_dim)

def test_encoder_block(config, hidden_states, batch_size):
    block = EncoderBlock(config)
    encout = block(hidden_states=hidden_states, return_attention=False)
    assert tuple(encout.shape) == (batch_size, config.num_objects, config.context_length, config.embed_dim)

def test_encoder(config, input_sequences, batch_size):
    encoder = Encoder(config)
    hidden_states = encoder(input_sequences, return_attention=False)
    assert tuple(hidden_states.shape) == (batch_size, config.num_objects, config.context_length, config.embed_dim)

def test_qgb(config, input_sequences, batch_size):
    encoder = Encoder(config)
    hidden_states = encoder(input_sequences, return_attention=False)
    qgb = QueryGenerationBlock(config)
    queries, encoder_states = qgb(hidden_states)
    assert tuple(encoder_states.shape) == (batch_size, config.num_objects, config.context_length, config.embed_dim)
    assert tuple(queries.shape) == (batch_size, config.num_objects, config.prediction_length, config.embed_dim)

def test_decoder_block(config, input_sequences, batch_size):
    encoder = Encoder(config)
    hidden_states = encoder(input_sequences, return_attention=False)
    qgb = QueryGenerationBlock(config)
    queries, encoder_states = qgb(hidden_states)
    decoder = DecoderBlock(config)
    hidden_states = decoder(queries, encoder_states)
    assert tuple(hidden_states.shape) == (batch_size, config.num_objects, config.prediction_length, config.embed_dim)

def test_decoder(config, input_sequences, batch_size):
    encoder = Encoder(config)
    hidden_states = encoder(input_sequences, return_attention=False)
    qgb = QueryGenerationBlock(config)
    queries, encoder_states = qgb(hidden_states)
    decoder = Decoder(config)
    hidden_states = decoder(queries, encoder_states)
    assert tuple(hidden_states.shape) == (batch_size, config.num_objects, config.prediction_length, config.embed_dim)
