import torch
from .attention import MultiHeadAttention, PositionWiseFeedForward
from .util import ResidualConnect, LayerNorm


class DecoderLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_attention_heads: int, dropout_prob: float):
        super().__init__()
        self.hidden_size: int = hidden_size
        self.intermediate_size: int = intermediate_size
        self.num_attention_heads: int = num_attention_heads
        self.dropout_prob: float = dropout_prob

        self.self_attention = MultiHeadAttention(self.hidden_size, self.num_attention_heads, self.dropout_prob)
        self.encoder_decoder_attention = MultiHeadAttention(self.hidden_size, self.num_attention_heads, self.dropout_prob)
        self.feed_forward = PositionWiseFeedForward(self.hidden_size, self.intermediate_size, self.dropout_prob)

        self.attention_residual = ResidualConnect(self.hidden_size, self.dropout_prob)
        self.memory_attention_residual = ResidualConnect(self.hidden_size, self.dropout_prob)
        self.feed_forward_residual = ResidualConnect(self.hidden_size, self.dropout_prob)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor,
                encoder_hidden_states: torch.Tensor, encoder_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.attention_residual(hidden_states, lambda x: self.self_attention(x, x, x, mask))
        hidden_states = self.memory_attention_residual(hidden_states, lambda x: self.encoder_decoder_attention(
            x, encoder_hidden_states, encoder_hidden_states, encoder_mask))
        return self.feed_forward_residual(hidden_states, self.feed_forward)


class DecoderStack(torch.nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, intermediate_size: int,
                 num_attention_heads: int, dropout_prob: float):
        super().__init__()
        self.num_layers: int = num_layers
        self.hidden_size: int = hidden_size
        self.intermediate_size: int = intermediate_size
        self.num_attention_heads: int = num_attention_heads
        self.dropout_prob: float = dropout_prob

        self.layers = torch.nn.ModuleList([
            DecoderLayer(self.hidden_size, self.intermediate_size, self.num_attention_heads, self.dropout_prob)
            for _ in range(self.num_layers)
        ])

        self.layer_norm = LayerNorm(self.hidden_size)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor,
                encoder_hidden_states: torch.Tensor, encoder_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask, encoder_hidden_states, encoder_mask)
        return self.layer_norm(hidden_states)
