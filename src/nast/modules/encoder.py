import torch
import torch.nn as nn

from torch import FloatTensor

from .attention import MultiheadAttention


class SpatialTemporalEncoderBlock(nn.Module):

    def __init__(self):...