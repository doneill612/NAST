from dataclasses import dataclass

@dataclass
class NastTransformerConfig:

    context_length: int
    prediction_length: int
    channels: int

    encoder_blocks: int = 2
    decoder_blocks: int = 2
    
    embed_dim: int = 32
    encoder_attn_heads: int = 4
    decoder_attn_heads: int = 4
    encoder_attn_dropout: float = 0.1
    decoder_attn_dropout: float = 0.1

    encoder_ff_expansion: int = 4
    encoder_ff_dropout: float = 0.1
    decoder_ff_expansion: int = 4
    decoder_ff_dropout: float = 0.1