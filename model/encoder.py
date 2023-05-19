import torch
from .attention import MultiHeadAttention, PositionWiseFeedForward
from .util import ResidualConnect, LayerNorm
from typing import Tuple
from copy import deepcopy


class EncoderLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_attention_heads: int, dropout_prob: float):
        super().__init__()
        self.hidden_size: int = hidden_size
        self.intermediate_size: int = intermediate_size
        self.num_attention_heads: int = num_attention_heads
        self.dropout_prob: float = dropout_prob

        self.self_attention = MultiHeadAttention(self.hidden_size, self.num_attention_heads, self.dropout_prob)
        self.residual_attention = ResidualConnect(self.hidden_size, self.dropout_prob)
        self.residual_feed_forward = deepcopy(self.residual_attention)
        self.feed_forward = PositionWiseFeedForward(self.hidden_size, self.intermediate_size, self.dropout_prob)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.residual_attention(hidden_states, lambda _: self.self_attention(_, _, _, mask))
        return self.residual_feed_forward(hidden_states, self.feed_forward)


class EncoderStack(torch.nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, intermediate_size: int,
                 num_attention_heads: int, dropout_prob: float):
        super().__init__()
        self.num_layers: int = num_layers
        self.hidden_size: int = hidden_size
        self.intermediate_size: int = intermediate_size
        self.num_attention_heads: int = num_attention_heads
        self.dropout_prob: float = dropout_prob

        self.layers = torch.nn.ModuleList([
            EncoderLayer(self.hidden_size, self.intermediate_size, self.num_attention_heads, self.dropout_prob)
            for _ in range(self.num_layers)
        ])
        self.layer_norm = LayerNorm(self.hidden_size)

    def forward(self, encoder_hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            encoder_hidden_states = layer(encoder_hidden_states, mask)
        return self.layer_norm(encoder_hidden_states)
