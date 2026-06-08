from datasets import load_dataset
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

def load_wikitext103(workflow_type: str = "train") -> list[str]:
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=workflow_type)
    raw = "\n".join(ds["text"])

    documents = []
    current: list[str] = []
    for line in raw.splitlines():
        if line.startswith(" = ") and not line.startswith(" = = "):
            if current:
                documents.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        documents.append("\n".join(current))
    return [doc for doc in documents if doc.strip()]

def create_dataloaders(
    batch_size: int = 5,
    max_length: int = 256,
    stride: int = 128,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    tokenizer = Tokenizer(model_name="gpt2")
    data_loaders = {}
    for split in ["train", "validation", "test"]:
        texts = load_wikitext103(split)
        dataset = LLMDataset(tokenizer, texts, max_length=max_length, stride=stride)
        data_loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            num_workers=num_workers,
        )
    return data_loaders
