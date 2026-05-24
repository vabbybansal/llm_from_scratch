from urllib.request import urlopen

from torch.utils.data import DataLoader

from llm_from_scratch.libs.llm_dataset import LLMDataset
from llm_from_scratch.libs.tokenizer import Tokenizer

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def load_tiny_shakespeare(url: str = TINY_SHAKESPEARE_URL) -> str:
    with urlopen(url) as response:
        return response.read().decode("utf-8")


def create_dataloader(
    texts: list[str] | None = None,
    batch_size: int = 5,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    if texts is None:
        texts = [load_tiny_shakespeare()]

    tokenizer = Tokenizer(model_name="gpt2")
    dataset = LLMDataset(tokenizer, texts, max_length=max_length, stride=stride)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
