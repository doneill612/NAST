import torch
import torch.nn as nn
import torch.nn.functional as functional

from torch import FloatTensor


class FeedForwardBlock(nn.Module):

    def __init__(self, embed_dim: int, expansion_factor: int, ff_dropout: float=0.0, bias: bool=False):
        super(FeedForwardBlock, self).__init__()
        self.ff_dropout = ff_dropout
        self.expand = nn.Linear(embed_dim, expansion_factor * embed_dim, bias=bias)
        self.contract = nn.Linear(expansion_factor * embed_dim, embed_dim, bias=bias)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, hidden_states: FloatTensor) -> FloatTensor:
        residual = hidden_states
        hidden_states = self.expand(hidden_states)
        hidden_states = self.contract(hidden_states)
        hidden_states = functional.dropout(hidden_states, p=self.ff_dropout, training=self.training)
        hidden_states += residual
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states