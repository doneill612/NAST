import math
import torch
import torch.nn as nn
import torch.nn.functional as functional

from torch import FloatTensor

class QueryGenerationBlock(nn.Module):
    
    def __init__(self, channels: int, prediction_length: int, embedding_dim: int, bias: bool=True):
        super(QueryGenerationBlock, self).__init__()
        self.channels = channels
        self.prediction_length = prediction_length
        self.embedding_dim = embedding_dim
        
        self.history_proj = nn.Linear(self.embedding_dim, 1, bias=bias)
        self.pos_proj = nn.Linear(self.embedding_dim, 1, bias=bias)
        
    def forward(self, encoder_outputs: FloatTensor) -> FloatTensor:
        
        history_scores = self.history_proj(encoder_outputs)
        history_scores = torch.relu(history_scores)
        
        position_encoding = self.get_position_encoding()
        position_scores = self.pos_proj(position_encoding)
        position_scores = torch.relu(position_scores)
        # position_scores: tensor of shape (batch_size, num_objects, prediction_length, 1)
        
       # Compute dot product of H(n) and P(n) to get weights for each object at each timestep
        weights = torch.bmm(history_scores.permute(0, 1, 3, 2), position_scores)
        # weights: tensor of shape (batch_size, num_objects, 1, prediction_length)

        # Apply the weights to the encoder_outputs to obtain the queries
        queries = torch.bmm(weights, encoder_outputs)
        # Reshape the queries tensor to remove the extra dimension
        queries = queries.squeeze(dim=2)
        
        return queries
        
    def get_position_encoding(self):
        position_encoding = torch.zeros(1, self.channels, self.prediction_length, self.embedding_dim)
        position = torch.arange(0, self.prediction_length).unsqueeze(0).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * (-math.log(10000.0) / self.embedding_dim))
        sin_term = torch.sin(position * div_term)
        cos_term = torch.cos(position * div_term)
        position_encoding[:, :, :, 0::2] = sin_term
        position_encoding[:, :, :, 1::2] = cos_term
        return position_encoding
