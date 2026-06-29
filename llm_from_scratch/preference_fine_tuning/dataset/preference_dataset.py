import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class OffPolicyDataset(Dataset):
    """
    Preference dataset for reward-model training. Each example is a (chosen, rejected) PAIR of full
    conversations (prompt + one assistant response each). Two key differences from SFTDataset:
      - NO -100 masking: the RM scores the *whole* sequence to a single scalar, so there are no
        per-token labels to mask.
      - NO input/target shift: the RM isn't doing next-token prediction, so __getitem__ just returns
        the raw token ids for each side. The learning signal is the Bradley-Terry loss comparing the
        two scalars (r_chosen vs r_rejected), built in the trainer.
    """
    def __init__(self, data, tokenizer, max_length):
        self.data = data            # already-loaded/filtered HF dataset; the factory owns loading/splitting
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _encode(self, messages):
        # full conversation (user + assistant) -> token ids. add_generation_prompt=False keeps the
        # assistant response in the sequence (we are SCORING the response, not generating it).
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, add_generation_prompt=False
        )
        # left-truncate: the differentiating response sits at the END, so keep the tail (same as SFT)
        return ids[-self.max_length:]

    def __getitem__(self, idx):
        row = self.data[idx]                          # tulu pref schema: row["chosen"], row["rejected"]
        chosen_ids = self._encode(row["chosen"])      # [user..<eot> assistant(preferred)..<eot>]
        rejected_ids = self._encode(row["rejected"])  # [user..<eot> assistant(dispreferred)..<eot>]
        return (torch.tensor(chosen_ids, dtype=torch.long),
                torch.tensor(rejected_ids, dtype=torch.long))


def _pad_side(seqs, pad_token_id, max_len):
    """
    Right-pad a list of 1-D id tensors to max_len AND build the matching attention mask.
    Why an attention mask here when SFT didn't need one: the RM pools the LAST REAL token
    (attention_mask.sum(1)-1) to get its scalar. SFT could skip the mask because its -100 targets +
    causal masking made pads harmless; the RM has no -100 fallback, so the mask is the ONLY record of
    where the real tokens end. Right-padding keeps that boundary at sum(mask)-1.
    """
    input_ids, attention_mask = [], []
    for s in seqs:
        pad_len = max_len - len(s)
        input_ids.append(torch.cat([s, torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        attention_mask.append(torch.cat([torch.ones(len(s), dtype=torch.long),
                                          torch.zeros(pad_len, dtype=torch.long)]))
    return torch.stack(input_ids), torch.stack(attention_mask)


def preference_collate(batch, pad_token_id, use_dynamic_padding=True, max_length=None):
    """
    Pad chosen and rejected INDEPENDENTLY and return a dict of four tensors.
    dynamic -> pad each side to that side's longest sequence in the batch (cheaper, variable shape).
    static  -> pad both sides to fixed max_length (constant shape -> no MPS allocator fragmentation).
    """
    chosen, rejected = zip(*batch)
    chosen_len = max(len(x) for x in chosen) if use_dynamic_padding else max_length
    rejected_len = max(len(x) for x in rejected) if use_dynamic_padding else max_length

    chosen_input_ids, chosen_attention_mask = _pad_side(chosen, pad_token_id, chosen_len)
    rejected_input_ids, rejected_attention_mask = _pad_side(rejected, pad_token_id, rejected_len)

    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": chosen_attention_mask,
        "rejected_input_ids": rejected_input_ids,
        "rejected_attention_mask": rejected_attention_mask,
    }


# --- filter_small: why we subset (same rationale as SFT's create_sft_dataloaders) ---
# A full Tulu-3 preference epoch is intractable on a Mac/MPS — and the RM does TWO forward passes per
# example (chosen + rejected), so it's even heavier than SFT. filter_small builds a small subset of
# SHORT, COMPLETE pairs so a run finishes in ~1 hour:
#   1. scan only a bounded window (train[:filter_scan]) so the filtering tokenization stays cheap
#      instead of tokenizing the whole mixture.
#   2. keep a pair only if BOTH chosen and rejected fit fully in max_length (no truncation): a pair is
#      only usable if neither side is cut, and shorter sequences = fewer tokens/step = faster + lower memory.
#   3. max_examples then caps the kept count, which bounds the number of training steps.
# Trade-off: biases the data toward short/simple tasks. The structural fix is LoRA (kills the
# optimizer-memory cost), not shrinking the dataset — see OVERVIEW "Memory & performance".
def create_preference_dataloaders(tokenizer, max_length=1024, batch_size=4,
                                  dataset_name="allenai/llama-3.1-tulu-3-8b-preference-mixture",
                                  val_fraction=0.01, num_workers=0,
                                  filter_small=False, max_examples=None, filter_scan=50000,
                                  use_dynamic_padding=True):
    if filter_small:
        # scan a bounded window and keep only pairs where BOTH sides fit fully in max_length (no
        # truncation) -> a small, fast subset of short complete pairs (biased toward simple tasks)
        data = load_dataset(dataset_name, split=f"train[:{filter_scan}]")
        def short_enough(ex):
            c = tokenizer.apply_chat_template(ex["chosen"], tokenize=True, return_dict=False, add_generation_prompt=False)
            r = tokenizer.apply_chat_template(ex["rejected"], tokenize=True, return_dict=False, add_generation_prompt=False)
            return len(c) <= max_length and len(r) <= max_length
        data = data.filter(short_enough)
    else:
        data = load_dataset(dataset_name, split="train")

    if max_examples:
        data = data.select(range(min(max_examples, len(data))))   # cap total count for a tractable run

    # carve validation off the end (the pref mixture ships only a "train" split)
    n_val = max(1, int(val_fraction * len(data)))                 # floor at 1 so validation is never empty
    train_data = data.select(range(len(data) - n_val))
    val_data = data.select(range(len(data) - n_val, len(data)))

    train_ds = OffPolicyDataset(train_data, tokenizer, max_length)
    val_ds = OffPolicyDataset(val_data, tokenizer, max_length)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    # no input/target shift for the RM, so static padding pads to the full max_length (not max_length-1)
    collate = partial(preference_collate, pad_token_id=pad_token_id,
                      use_dynamic_padding=use_dynamic_padding, max_length=max_length)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            collate_fn=collate, num_workers=num_workers),
        "validation": DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate, num_workers=num_workers),
    }
