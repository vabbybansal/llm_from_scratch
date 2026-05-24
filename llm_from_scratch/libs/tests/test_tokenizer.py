import unittest
from llm_from_scratch.libs.tokenizer import Tokenizer


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


if __name__ == "__main__":
    unittest.main()
