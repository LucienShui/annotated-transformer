import unittest
import torch


class TestInference(unittest.TestCase):
    def test_inference(self):
        from model.transformer import Transformer

        transformer = Transformer(128, 256, 4, 256, 1024, 8, 0.1)
        transformer.eval()

        input_ids = torch.LongTensor([list(range(1, 1 + 10))])
        mask = torch.ones((1, 1, 10))

        encoder_hidden_states = transformer.encode(input_ids, mask)
        decoder_input_ids = torch.zeros(1, 1).long()

        # for i in range(10):
        #     decoder_hidden_states = transformer.decode(decoder_input_ids, )


if __name__ == '__main__':
    unittest.main()
