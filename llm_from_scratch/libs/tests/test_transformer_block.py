import unittest

import torch

from llm_from_scratch.libs.transformer_block import TransformerBlock


class TestTransformerBlock(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.batch_size = 2
        self.num_tokens = 5
        self.emb_dim = 8
        self.context_length = 8

        self.block = TransformerBlock(
            d_in=self.emb_dim,
            d_out=self.emb_dim,
            context_length=self.context_length,
            dropout=0.0,
            n_heads=2,
        )
        self.block.eval()
        self.x = torch.randn(self.batch_size, self.num_tokens, self.emb_dim)

    def test_output_shape(self):
        output = self.block(self.x)

        self.assertEqual(output.shape, self.x.shape)

    def test_gradients_flow(self):
        self.block.train()
        x = self.x.detach().requires_grad_(True)
        output = self.block(x)
        output.sum().backward()

        self.assertIsNotNone(x.grad)

    def test_deterministic_in_eval(self):
        out_a = self.block(self.x)
        out_b = self.block(self.x)

        torch.testing.assert_close(out_a, out_b)

    def test_d_in_must_equal_d_out(self):
        with self.assertRaises(AssertionError):
            TransformerBlock(
                d_in=8,
                d_out=16,
                context_length=self.context_length,
                dropout=0.0,
                n_heads=2,
            )


if __name__ == "__main__":
    unittest.main()
