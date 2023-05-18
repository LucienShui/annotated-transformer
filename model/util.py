import torch
from typing import Callable


class LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        self.eps: float = eps
        self.weight = torch.nn.Parameter(torch.ones(self.hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(self.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean: torch.Tensor = x.mean(-1, keepdim=True)
        std: torch.Tensor = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class ResidualConnect(torch.nn.Module):
    def __init__(self, hidden_size: int, dropout_rate: float):
        super().__init__()
        self.hidden_size: int = hidden_size
        self.dropout_rate: float = dropout_rate
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = torch.nn.Dropout()

    def forward(self, x: torch.Tensor, layer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        return x + self.dropout(layer(self.layer_norm(x)))
