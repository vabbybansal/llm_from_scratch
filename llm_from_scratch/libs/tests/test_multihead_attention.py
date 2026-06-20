import unittest

import torch

from llm_from_scratch.libs.multihead_attention import MultiHeadAttention


class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.batch_size = 2
        self.num_tokens = 5
        self.d_in = 4
        self.d_out = 8
        self.n_heads = 2
        self.context_length = 8
        self.dropout = 0.0

        self.attn = MultiHeadAttention(
            d_in=self.d_in,
            d_out=self.d_out,
            context_length=self.context_length,
            dropout=self.dropout,
            n_heads=self.n_heads,
        )
        self.attn.eval()
        self.x = torch.randn(self.batch_size, self.num_tokens, self.d_in)

    def test_output_shape(self):
        output = self.attn(self.x)

        self.assertEqual(output.shape, (self.batch_size, self.num_tokens, self.d_out))

    def test_d_out_must_be_divisible_by_n_heads(self):
        with self.assertRaises(AssertionError):
            MultiHeadAttention(
                d_in=self.d_in,
                d_out=7,
                context_length=self.context_length,
                dropout=self.dropout,
                n_heads=2,
            )

    def test_causal_mask_ignores_future_tokens(self):
        x_modified = self.x.clone()
        x_modified[:, 3:, :] = torch.randn_like(x_modified[:, 3:, :])

        out_original = self.attn(self.x)
        out_modified = self.attn(x_modified)

        torch.testing.assert_close(out_original[:, :3, :], out_modified[:, :3, :])

    def test_future_positions_can_change_when_past_changes(self):
        x_modified = self.x.clone()
        x_modified[:, 0, :] = torch.randn(self.d_in)

        out_original = self.attn(self.x)
        out_modified = self.attn(x_modified)

        self.assertFalse(torch.allclose(out_original[:, 1:, :], out_modified[:, 1:, :]))

    def test_gradients_flow(self):
        self.attn.train()
        x = self.x.detach().requires_grad_(True)
        output = self.attn(x)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(self.attn.c_attn.weight.grad)

    def test_dropout_zero_is_deterministic_in_eval(self):
        out_a = self.attn(self.x)
        out_b = self.attn(self.x)

        torch.testing.assert_close(out_a, out_b)


if __name__ == "__main__":
    unittest.main()
