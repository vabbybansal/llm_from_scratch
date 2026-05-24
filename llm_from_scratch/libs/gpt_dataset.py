import torch
from torch.utils.data import Dataset
from llm_from_scratch.libs.tokenizer import Tokenizer

class GPTDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, documents:list[str], max_length:int, stride:int):

        self.input_ids = []
        self.target_ids = []
        token_ids = []

        for document in documents:
            token_ids.extend(tokenizer.encode(document))
            token_ids.append(tokenizer.get_eos_token_id())  # append EOS token between documents

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i+max_length]
            target_chunk = token_ids[i+1 : i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
        