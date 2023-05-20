import unittest
import torch


def subsequent_mask(size):
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).long()
    return mask == 0


class TestInference(unittest.TestCase):
    def test_inference(self):
        from model.transformer import TransformerWithLMHead

        transformer = TransformerWithLMHead(128, 256, 4, 256, 1024, 8, 0.1)
        transformer.eval()

        input_ids = torch.LongTensor([list(range(1, 1 + 10))])
        mask = torch.ones((1, 1, 10))

        encoder_hidden_states = transformer.encode(input_ids, mask)
        decoder_input_ids = torch.zeros(1, 1).long()

        for i in range(20):
            prob: torch.Tensor = transformer.decode(
                decoder_input_ids, subsequent_mask(decoder_input_ids.size(1)), encoder_hidden_states, mask)
            output_ids = torch.argmax(prob, dim=1).unsqueeze(1)
            decoder_input_ids = torch.cat([decoder_input_ids, output_ids], dim=1)
            print(decoder_input_ids)


if __name__ == '__main__':
    unittest.main()
