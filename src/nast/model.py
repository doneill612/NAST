import torch
import torch.nn as nn
import torch.nn.functional as functional

from typing import Union, Tuple, Optional
from torch import FloatTensor

from config import NastTransformerConfig
from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.qgb import QueryGenerationBlock


class NastTransformer(nn.Module):

    def __init__(self, config: NastTransformerConfig):
        super(NastTransformer, self).__init__()
        self.encoder = Encoder(config)
        self.qgb = QueryGenerationBlock(config)
        self.decoder = Decoder(config)
