import torch
from torch import Tensor
from .attention import MultiHeadAttention, PositionWiseFeedForward
from .util import ResidualConnect
from copy import deepcopy


class Decoder(torch.nn.Module):
    def __init__(self, hidden_size: int, big_hidden_size: int, head_count: int, dropout_rate: float):
        super().__init__()
        self.hidden_size: int = hidden_size
        self.big_hidden_size: int = big_hidden_size
        self.head_count: int = head_count
        self.dropout_rate: float = dropout_rate

        self.attention = MultiHeadAttention(self.hidden_size, self.head_count, self.dropout_rate)
        self.memory_attention = deepcopy(self.attention)
        self.feed_forward = PositionWiseFeedForward(self.hidden_size, self.big_hidden_size, self.dropout_rate)

        self.attention_residual = ResidualConnect(self.hidden_size, self.dropout_rate)
        self.memory_attention_residual = deepcopy(self.attention_residual)
        self.feed_forward_residual = deepcopy(self.attention_residual)

    def forward(self, x: Tensor, memory: Tensor, mask: Tensor):
        pass
