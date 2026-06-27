import unittest
from llm_from_scratch.pretraining.data.tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):
    ENCODING = "gpt2"

    def setUp(self):
        self.tokenizer = Tokenizer(self.ENCODING)

    def test_encode_returns_list_of_ints(self):
        tokens = self.tokenizer.encode("hello world")

        self.assertIsInstance(tokens, list)
        self.assertTrue(tokens)
        self.assertTrue(all(isinstance(token, int) for token in tokens))

    def test_decode_roundtrip(self):
        text = "Hello, world!"

        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_encode_empty_string(self):
        self.assertEqual(self.tokenizer.encode(""), [])

    def test_decode_empty_list(self):
        self.assertEqual(self.tokenizer.decode([]), "")

    def test_eos_token_id_is_int(self):
        self.assertIsInstance(self.tokenizer.get_eos_token_id(), int)

    def test_eos_token_roundtrip(self):
        eos_id = self.tokenizer.get_eos_token_id()

        self.assertEqual(self.tokenizer.decode([eos_id]), self.tokenizer.get_eos_string())


if __name__ == "__main__":
    unittest.main()
