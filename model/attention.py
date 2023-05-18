import math

import torch
from typing import Tuple


class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_size: int, big_hidden_size: int, dropout_rate: float = 0.1):
        super().__init__()
        self.hidden_size: int = hidden_size
        self.big_hidden_size: int = big_hidden_size
        self.dropout_rate: float = dropout_rate

        self.w_1 = torch.nn.Linear(self.hidden_size, self.big_hidden_size)
        self.w_2 = torch.nn.Linear(self.big_hidden_size, self.hidden_size)
        self.dropout = torch.nn.Dropout(self.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.w_1(x).relu()))


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_size: int, head_count: int, dropout_rate: float = 0.1):
        super().__init__()
        self.head_count: int = head_count
        self.hidden_size: int = hidden_size
        self.dropout_rate: float = dropout_rate

        assert self.hidden_size % self.head_count == 0

        self.d_k: int = self.hidden_size // self.head_count
        self.linear_list = torch.nn.ModuleList([torch.nn.Linear(self.hidden_size, self.hidden_size) for _ in range(3)])
        self.output_layer = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout: torch.nn.Dropout = torch.nn.Dropout(self.dropout_rate)
        self.eps: float = 1e-9
        self.attention_result = ...

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores: torch.Tensor = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = torch.masked_fill(scores, mask == 0, self.eps)
        p_attention: torch.Tensor = self.dropout(torch.softmax(scores, dim=-1))
        return torch.matmul(p_attention, v), p_attention

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            mask: torch.Tensor = mask.unsqueeze(1)
        batch_size: int = q.size(0)

        q, k, v = [
            w(x).view(batch_size, -1, self.head_count, self.hidden_size, self.d_k).transpose(1, 2)
            for w, x in zip(self.linear_list, (q, k, v))
        ]
        x, self.attention_result = self.attention(q, k, v, mask=mask)
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        return self.output_layer(x)
