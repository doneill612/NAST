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
from nast.config import NastTransformerConfig

@pytest.fixture
def config():
    return NastTransformerConfig(
        context_length=15,
        prediction_length=5,
        channels=3
    )

@pytest.fixture
def timeseries(config):
    ts = np.zeros(shape=(config.channels, config.context_length))
    ts = FloatTensor(config.channels, config.context_length)
    return FloatTensor(ts)

@pytest.fixture
def batch_size():
    return 12

def test_positional_encoding_table(config, timeseries):
    sequence_length = config.context_length
    embed_dim = config.embed_dim
    channels = config.channels
    
    enc_table = positional_encoding_table(sequence_length, embed_dim)
    
    assert enc_table.size(0) == sequence_length, enc_table.size()
    assert enc_table.size(1) == embed_dim, enc_table.size()

    pos_embedding = nn.Embedding.from_pretrained(enc_table, freeze=True)
    ts_embedding = nn.Sequential(nn.Conv1d(channels, embed_dim, kernel_size=1), nn.LayerNorm((embed_dim, sequence_length)))
    pos = torch.arange(0, sequence_length, dtype=torch.long)
    timeseries = timeseries.expand(12, *timeseries.size())
    timeseries = ts_embedding(timeseries).permute(0, 2, 1) + pos_embedding(pos)

    assert timeseries.size(1) == sequence_length, timeseries.size()
    assert timeseries.size(2) == embed_dim, timeseries.size()