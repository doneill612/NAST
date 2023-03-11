import torch
import torch.nn as nn
import torch.nn.functional as functional

from typing import Union, Tuple, Optional
from torch import FloatTensor

from .attention import MultiheadAttention, positional_encoding_table
from .mlp import FeedForwardBlock
from ..config import NastTransformerConfig


class Decoder(nn.Module):
    ...

class DecoderBlock(nn.Module):
    ...