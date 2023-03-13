import torch.nn as nn

from torch import FloatTensor

from .config import NastTransformerConfig
from .modules.encoder import Encoder
from .modules.decoder import Decoder
from .modules.qgb import QueryGenerationBlock


class NastTransformerBase(nn.Module):

    def __init__(self, config: NastTransformerConfig):
        super(NastTransformerBase, self).__init__()
        self.encoder = Encoder(config)
        self.qgb = QueryGenerationBlock(config)
        self.decoder = Decoder(config)

    def forward(self, input_sequences: FloatTensor):
        encout = self.encoder(input_sequences, return_attention=False)
        decoder_queries, encout = self.qgb(encout)
        hidden_states = self.decoder(decoder_queries, encout)
        return hidden_states
        
