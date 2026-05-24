import unittest

from llm_from_scratch.libs.llm_dataset import LLMDataset
from llm_from_scratch.libs.tokenizer import Tokenizer


class TestLLMDataset(unittest.TestCase):
    ENCODING = "gpt2"

    def setUp(self):
        self.tokenizer = Tokenizer(self.ENCODING)
        self.documents = ["hello world", "foo bar baz"]
        self.max_length = 4
        self.stride = 2

    def _expected_token_ids(self) -> list[int]:
        token_ids = []
        for document in self.documents:
            token_ids.extend(self.tokenizer.encode(document))
            token_ids.append(self.tokenizer.get_eos_token_id())
        return token_ids

    def test_len_matches_sliding_window_count(self):
        token_ids = self._expected_token_ids()
        expected_len = len(range(0, len(token_ids) - self.max_length, self.stride))
        dataset = LLMDataset(
            self.tokenizer, self.documents, self.max_length, self.stride
        )

        self.assertEqual(len(dataset), expected_len)

    def test_getitem_returns_tensors_with_correct_shape(self):
        dataset = LLMDataset(
            self.tokenizer, self.documents, self.max_length, stride=1
        )
        input_ids, target_ids = dataset[0]

        self.assertEqual(input_ids.shape, (self.max_length,))
        self.assertEqual(target_ids.shape, (self.max_length,))

    def test_all_windows_match_expected_token_ids(self):
        token_ids = self._expected_token_ids()
        dataset = LLMDataset(
            self.tokenizer, self.documents, self.max_length, self.stride
        )

        for idx, i in enumerate(
            range(0, len(token_ids) - self.max_length, self.stride)
        ):
            input_ids, target_ids = dataset[idx]
            print(input_ids)
            self.assertEqual(input_ids.tolist(), token_ids[i : i + self.max_length])
            self.assertEqual(
                target_ids.tolist(), token_ids[i + 1 : i + self.max_length + 1]
            )

    def test_inserts_eos_between_documents(self):
        documents = ["hello", "world"]
        token_ids = []
        for document in documents:
            token_ids.extend(self.tokenizer.encode(document))
            token_ids.append(self.tokenizer.get_eos_token_id())

        eos_id = self.tokenizer.get_eos_token_id()
        first_doc_end = len(self.tokenizer.encode(documents[0]))

        self.assertEqual(token_ids[first_doc_end], eos_id)

    def test_empty_dataset_when_sequence_shorter_than_max_length(self):
        dataset = LLMDataset(self.tokenizer, ["a"], max_length=100, stride=1)

        self.assertEqual(len(dataset), 0)


if __name__ == "__main__":
    unittest.main()
