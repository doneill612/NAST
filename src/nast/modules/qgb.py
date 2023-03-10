import math
import torch
import torch.nn as nn
import torch.nn.functional as functional

from torch import FloatTensor

from .attention import positional_encoding_table


class QueryGenerationBlock(nn.Module):
    
    def __init__(self, channels: int, prediction_length: int, embed_dim: int, bias: bool=True):
        super(QueryGenerationBlock, self).__init__()
        self.channels = channels
        self.prediction_length = prediction_length
        self.embedding_dim = embed_dim

        self.pos_embed = nn.Embedding.from_pretrained(
            positional_encoding_table(prediction_length, embed_dim),
            freeze=True
        )
        
        self.history_proj = nn.Linear(self.embedding_dim, 1, bias=bias)
        self.pos_proj = nn.Linear(self.embedding_dim, 1, bias=bias)
        
    def forward(self, encoder_outputs: FloatTensor, return_encoder_outputs: bool=True) -> FloatTensor:
        # shape(encoder_outputs) = (batch_size, channels, seq_len, embed_dim)
        history_scores = self.history_proj(encoder_outputs)
        history_scores = torch.relu(history_scores)
        
        positions = torch.arange(0, self.prediction_length, dtype=torch.long).to(encoder_outputs.device)
        position_encoding = self.pos_embed(positions)
        position_scores = self.pos_proj(position_encoding).transpose(0, 1).contiguous()
        position_scores = torch.relu(position_scores)

        
        weights = torch.matmul(history_scores, position_scores)
        queries = torch.matmul(weights.mT, encoder_outputs)

        if return_encoder_outputs:
            return queries, encoder_outputs
        return queries