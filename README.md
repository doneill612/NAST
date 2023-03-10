# NAST
## Non-Autoregressive Spatial-Temporal Transformer

This is a `torch` implementation of the transformer architecture proposed in [Chen et al. 2021](https://arxiv.org/pdf/2102.05624v1.pdf) aimed to adapt the concept of spatial-temporal attention to multivariate time-series forecasting. It provides a way to implement the network modularly, detaching the forecasting network heads from the hidden state ouput.

Design choices inspired heavily by [Vaswani et al.](https://github.com/jadore801120/attention-is-all-you-need-pytorch) and the [`transformers` library](https://huggingface.co/docs/transformers/index).