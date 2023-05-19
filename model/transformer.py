import torch
from .encoder import EncoderStack
from .decoder import DecoderStack


class Transformer(torch.nn.Module):
    def __init__(self, max_position_embeddings: int, vocab_size: int, num_layers: int, hidden_size: int,
                 intermediate_size: int, num_attention_heads: int, dropout_prob: float):
        super().__init__()
        self.max_position_embeddings: int = max_position_embeddings
        self.vocab_size: int = vocab_size
        self.num_layers: int = num_layers
        self.hidden_size: int = hidden_size
        self.intermediate_size: int = intermediate_size
        self.num_attention_heads: int = num_attention_heads
        self.dropout_prob: float = dropout_prob

        self.embedding = torch.nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embedding = torch.nn.Embedding(self.max_position_embeddings, self.hidden_size)

        self.encoder = EncoderStack(self.num_layers, self.hidden_size, self.intermediate_size,
                                    self.num_attention_heads, self.dropout_prob)
        self.decoder = DecoderStack(self.num_layers, self.hidden_size, self.intermediate_size,
                                    self.num_attention_heads, self.dropout_prob)

    def forward(self, encoder_input_ids: torch.Tensor, encoder_mask: torch.Tensor,
                decoder_input_ids: torch.Tensor, decoder_mask: torch.Tensor):
        return self.decode(decoder_input_ids, decoder_mask, self.encoder(encoder_input_ids, encoder_mask), encoder_mask)

    def encode(self, encoder_input_ids: torch.Tensor, encoder_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.embedding(encoder_input_ids), encoder_mask)

    def decode(self, input_dis: torch.Tensor, mask: torch.Tensor,
               encoder_hidden_states: torch.Tensor, encoder_mask: torch.Tensor) -> torch.Tensor:
        return self.decoder(input_dis, mask, encoder_hidden_states, encoder_mask)


class TransformerWithLMHead(torch.nn.Module):
    def __init__(self, max_position_embeddings: int, vocab_size: int, num_layers: int, hidden_size: int,
                 intermediate_size: int, num_attention_heads: int, dropout_prob: float):
        super().__init__()
        self.max_position_embeddings: int = max_position_embeddings
        self.vocab_size: int = vocab_size
        self.num_layers: int = num_layers
        self.hidden_size: int = hidden_size
        self.intermediate_size: int = intermediate_size
        self.num_attention_heads: int = num_attention_heads
        self.dropout_prob: float = dropout_prob

        self.transformer = Transformer(self.max_position_embeddings, self.vocab_size, self.num_layers, self.hidden_size,
                                       self.intermediate_size, self.num_attention_heads, self.dropout_prob)
        self.lm_head = torch.nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, encoder_input_ids: torch.Tensor, encoder_mask: torch.Tensor,
                decoder_input_ids: torch.Tensor, decoder_mask: torch.Tensor):
        hidden_states: torch.Tensor = self.transformer(encoder_input_ids, encoder_mask, decoder_input_ids, decoder_mask)
        logits: torch.Tensor = self.lm_head(hidden_states)

