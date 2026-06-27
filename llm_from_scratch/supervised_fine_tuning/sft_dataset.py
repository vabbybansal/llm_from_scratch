import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, ignore_token=-100):
        # data is an already-loaded (and possibly filtered) HF dataset; the factory below owns loading/splitting
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_token = ignore_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]["messages"]  # https://huggingface.co/datasets/allenai/tulu-3-sft-mixture/viewer/default/train?row=14

        # full conversation: used as input_ids
        # tokenize returns the tokens
        # add_generation_prompt False does not add any <assisstant> token to indicate the model for generation
        # <user>What is the capital of India<eot><user><assisstant>Delhi<eot>
        full_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=False)

        # prompt only (no last assistant turn): used to find where response starts, hence we mark with [:-1] which skips the last message. add generation prompt True adds the <assisstant> token which indicates the model for generation.
        # <user>What is the capital of India<eot><user><assisstant>
        prompt_ids = self.tokenizer.apply_chat_template(messages[:-1], tokenize=True, return_dict=False, add_generation_prompt=True)

        # mask instruction tokens, keep only response tokens in labels (aligned to full_ids)
        # <MASK><MASK><MASK><MASK><MASK><MASK><MASK><MASK><MASK><MASK><MASK><MASK>Delhi<eot>
        labels = [self.ignore_token] * len(prompt_ids) + full_ids[len(prompt_ids):]

        # truncate from the LEFT to max_length before shifting, so the response (at the end) is preserved
        full_ids = full_ids[-self.max_length:]
        labels = labels[-self.max_length:]

        # Option B: shift in the dataset. input_ids[i] predicts target[i].
        # target = labels shifted left by 1, so position i's input predicts the next token.
        input_ids = full_ids[:-1]                   # <user>What is the capital of India<eot><user> <assisstant>  Delhi
        target = labels[1:]                         # <MASK><MASK><MASK><MASK><MASK><MASK><MASK.SK> Delhi         <eot>

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def sft_collate(batch, pad_token_id, ignore_token=-100, use_dynamic_padding=True, max_length=None):
    '''
    Pads a batch and stacks it.
    use_dynamic_padding=True  -> pad to the batch's longest sequence (cheaper compute, but every batch is a
                                 different shape, which fragments the GPU caching allocator over time).
    use_dynamic_padding=False -> static: pad every batch to fixed `max_length`, so all batches share one
                                 shape -> no fragmentation, at the cost of wasted compute on pad tokens.
    Right-padding is safe for a causal LM without an attention mask: real tokens are causally masked from
    future pad tokens, and pad positions in the target are -100 so they contribute no loss.
    '''

    # <user>What is the capital of India<eot><user> <assisstant>  Delhi         => Input
    # <MASK><MASK><MASK><MASK><MASK><MASK><MASK.SK> Delhi         <eot>         => Target
    # <user>Hi<eot><user>  <assisstant> Hello                                   => Input
    # <MASK><MASK><M.MASK> Hello        <eot>                                   => Target

    # Becomes =>

    # <user>What is the capital of India<eot><user> <assisstant>  Delhi         => Input
    # <MASK><MASK><MASK><MASK><MASK><MASK><MASK.SK> Delhi         <eot>         => Target
    # <user>Hi<eot><user>  <assisstant> Hello <PAD> <PAD> <PAD> <PAD>           => Input
    # <MASK><MASK><M.MASK> Hello        <eot> <-100><-100><-100><-100>          => Target
    input_ids, targets = zip(*batch)
    # dynamic -> longest in this batch; static -> fixed max_length so every batch is the same shape
    max_len = max(len(x) for x in input_ids) if use_dynamic_padding else max_length

    padded_inputs, padded_targets = [], []
    for inp, tgt in zip(input_ids, targets):
        pad_len = max_len - len(inp)
        # as seen in example, add padding to the right. If we added this to the left, actual tokens would attend to them corrupting them (attention always looks to the left tokens)
        # inference on the otherhand needs padding to the left since we need the same final token towards the end and gen happens at the end. We do pass attention mask to handle padding tokens since otherwise the attention will get corrupted
        padded_inputs.append(torch.cat([inp, torch.full((pad_len,), pad_token_id, dtype=torch.long)]))  
        # as seen in example, with ignore tokens, these tokens in target for the input padding tokens are ignored from cross entropy loss which makes sense
        padded_targets.append(torch.cat([tgt, torch.full((pad_len,), ignore_token, dtype=torch.long)]))  # ignore token => -100

    return torch.stack(padded_inputs), torch.stack(padded_targets)


def create_sft_dataloaders(tokenizer, max_length=1024, batch_size=4,
                           dataset_name="allenai/tulu-3-sft-mixture", val_fraction=0.01,
                           num_workers=0, filter_small=False, max_examples=None, filter_scan=50000,
                           use_dynamic_padding=True):
    if filter_small:
        # scan a bounded window and keep only examples that fit fully in max_length (no truncation) —
        # gives a small, fast, low-memory subset of short complete conversations (biased toward simple tasks)
        data = load_dataset(dataset_name, split=f"train[:{filter_scan}]")
        def short_enough(ex):
            ids = tokenizer.apply_chat_template(ex["messages"], tokenize=True, return_dict=False, add_generation_prompt=False)
            return len(ids) <= max_length
        data = data.filter(short_enough)
    else:
        data = load_dataset(dataset_name, split="train")

    if max_examples:
        data = data.select(range(min(max_examples, len(data))))   # cap total count for a tractable run

    # carve validation off the end (tulu-3-sft-mixture ships only a "train" split)
    n_val = max(1, int(val_fraction * len(data)))                 # floor at 1 so validation is never empty
    train_data = data.select(range(len(data) - n_val))
    val_data = data.select(range(len(data) - n_val, len(data)))

    train_ds = SFTDataset(train_data, tokenizer, max_length)
    val_ds = SFTDataset(val_data, tokenizer, max_length)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    # post-shift inputs are at most max_length-1 long; that's the fixed target for static padding
    collate = partial(sft_collate, pad_token_id=pad_token_id,
                      use_dynamic_padding=use_dynamic_padding, max_length=max_length - 1)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            collate_fn=collate, num_workers=num_workers),
        "validation": DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=collate, num_workers=num_workers),
    }