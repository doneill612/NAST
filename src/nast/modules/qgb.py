import torch
import torch.nn as nn

from torch import FloatTensor

from .attention import positional_encoding_table
from ..config import NastTransformerConfig


class QueryGenerationBlock(nn.Module):
    """The query generation block as described in Chen et al. 2021.

    This block sits between the encoder and decoder blocks of the transformer.
    It serves the purpose of using the output hidden states of the encoder block
    to produce queries for the decoder in a single step, allowing the model to 
    perform parallel decoding.

    Given encoder output hidden states `encoder_ouptus` of shape `(b, c, th, e)`, produce a 
    history score tensor of shape `(b th c)` representing the temporal influence of 
    the input sequence (hidden states) on the queries. Use a positional 
    embedding table to produce a positonal encoding of shape `(tf, 1)` which 
    will represent spatial influence on the queries. Finally, synthesize the 
    temporal and spatial influences via a `matmul` operation, resulting in joint 
    influence `I` and generate queries of shape `(b c tf, e)` by performing
    `torch.matmul(I.mT, encoder_outputs)`.
    """
    def __init__(self, config: NastTransformerConfig, bias: bool=True):
        """Constructs a new query generation block.

        Args
        ----
            `prediction_length` (`int`)
                The length of the queries to produce, also corresponds
                to the length of the predicted time series in time series
                forecasting tasks.
            `embed_dim` (`int`)
                Embedding dimension of hidden states in the encoder and 
                decoder layers
            `bias` (`bool`, optional)
                Whether or not to use bias on the linear projection layers
        """
        super(QueryGenerationBlock, self).__init__()
        self.config = config
        self.pos_embed = nn.Embedding.from_pretrained(
            positional_encoding_table(config.prediction_length, config.embed_dim),
            freeze=True
        )
        self.history_proj = nn.Linear(config.embed_dim, 1, bias=bias)
        self.pos_proj = nn.Linear(config.embed_dim, 1, bias=bias)
        
    def forward(self, encoder_outputs: FloatTensor, return_encoder_outputs: bool=True) -> FloatTensor:
        # shape(encoder_outputs) = (batch_size, channels, seq_len, embed_dim)
        history_scores = self.history_proj(encoder_outputs)
        history_scores = torch.relu(history_scores)
        
        positions = torch.arange(0, self.config.prediction_length, dtype=torch.long).to(encoder_outputs.device)
        position_encoding = self.pos_embed(positions)
        position_scores = self.pos_proj(position_encoding).transpose(0, 1).contiguous()
        position_scores = torch.relu(position_scores)

        
        weights = torch.matmul(history_scores, position_scores)
        queries = torch.matmul(weights.mT, encoder_outputs)

        if return_encoder_outputs:
            # return the queries and the encoder outputs for cross attention
            return queries, encoder_outputs
        return queries