import unittest

import torch

from llm_from_scratch.libs.gpt import GPT


class TestGPT(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.batch_size = 2
        self.seq_len = 5
        self.vocab_size = 100
        self.d_in = 8
        self.context_length = 8
        self.n_layers = 2
        self.n_heads = 2

        self.model = GPT(
            vocab_size=self.vocab_size,
            n_layers=self.n_layers,
            d_in=self.d_in,
            context_length=self.context_length,
            dropout=0.0,
            n_heads=self.n_heads,
        )
        self.model.eval()
        self.token_ids = torch.randint(
            0, self.vocab_size, (self.batch_size, self.seq_len)
        )

    def test_output_shape(self):
        logits = self.model(self.token_ids)

        self.assertEqual(
            logits.shape, (self.batch_size, self.seq_len, self.vocab_size)
        )

    def test_gradients_flow(self):
        self.model.train()
        logits = self.model(self.token_ids)
        logits.sum().backward()

        self.assertIsNotNone(self.model.token_emb.weight.grad)
        self.assertIsNotNone(self.model.out_head.weight.grad)

    def test_deterministic_in_eval(self):
        out_a = self.model(self.token_ids)
        out_b = self.model(self.token_ids)

        torch.testing.assert_close(out_a, out_b)

    def test_tied_weights_share_embedding(self):
        self.assertIs(self.model.out_head.weight, self.model.token_emb.weight)

    def test_raises_when_seq_len_exceeds_context_length(self):
        long_ids = torch.randint(
            0, self.vocab_size, (self.batch_size, self.context_length + 1)
        )

        with self.assertRaises(IndexError):
            self.model(long_ids)


if __name__ == "__main__":
    unittest.main()
