import torch
from .attention import MultiHeadAttention, PositionWiseFeedForward
from .util import ResidualConnect
from typing import Tuple


class Encoder(torch.nn.Module):
    def __init__(self, hidden_size: int, big_hidden_size: int, head_count: int, dropout_rate: float):
        super().__init__()
        self.hidden_size: int = hidden_size
        self.big_hidden_size: int = big_hidden_size
        self.head_count: int = head_count
        self.dropout_rate: float = dropout_rate
        self.attention = MultiHeadAttention(self.hidden_size, self.head_count, self.dropout_rate)
        self.residual_attention = ResidualConnect(self.hidden_size, self.dropout_rate)
        self.residual_feed_forward = ResidualConnect(self.hidden_size, self.dropout_rate)
        self.feed_forward = PositionWiseFeedForward(self.hidden_size, self.big_hidden_size, self.dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.residual_attention(x, lambda _: self.attention(_, _, _, mask))
        return self.residual_feed_forward(x, self.feed_forward), mask



