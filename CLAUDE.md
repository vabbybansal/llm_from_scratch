# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install (editable):**
```bash
pip install -e .
```

**Run tests:**
```bash
python -m unittest discover -s llm_from_scratch/ -v
```

**Run a single test file:**
```bash
python -m pytest llm_from_scratch/scripts/test_dataloader.py
# or
python llm_from_scratch/scripts/test_dataloader.py
```

**Check worker/GPU environment (requires Ray cluster):**
```bash
python llm_from_scratch/scripts/test_worker_config.py
```

## Architecture

This is an educational implementation of a GPT-style language model built from primitives using PyTorch and tiktoken. The code lives in `llm_from_scratch/libs/` and is assembled bottom-up:

```
Tokenizer (tiktoken/gpt2 encoding)
  └─> LLMDataset (sliding-window token pairs for next-token prediction)
        └─> create_dataloader (wraps Dataset in a PyTorch DataLoader)

MultiHeadAttention  ─┐
FeedForward         ─┤─> TransformerBlock ─> GPT (full model)
LayerNorm (built-in) ┘
```

**Key design details:**

- `MultiHeadAttention` (`libs/multihead_attention.py`): implements causal (decoder-only) attention via an upper-triangular mask registered as a buffer. Projects Q/K/V with separate `nn.Linear` layers, splits by heads, computes scaled dot-product attention, then projects output.

- `TransformerBlock` (`libs/transformer_block.py`): uses **pre-norm** (LayerNorm before attention and FF, not after), with residual connections wrapping each sub-layer. Requires `d_in == d_out`.

- `FeedForward` (`libs/feedforward.py`): two-layer MLP with 4× expansion and GELU activation.

- `GPT` (`libs/gpt.py`): combines learned token + positional embeddings, a stack of `TransformerBlock`s, a final LayerNorm, and a linear output head. Supports optional weight tying between the token embedding and output projection.

- `LLMDataset` (`libs/llm_dataset.py`): sliding-window approach over concatenated documents (joined by EOS tokens). Each sample is `(input_ids[i:i+max_length], input_ids[i+1:i+max_length+1])`.

- `Tokenizer` (`libs/tokenizer.py`): thin wrapper around tiktoken using `gpt2` encoding by default.

**Notebooks** in `llm_from_scratch/scripts/` explore concepts interactively: `self_attention.ipynb`, `layer_norm.ipynb`, and `gpt.ipynb` (includes autoregressive generation visualization).

## Testing

When writing unit tests, write only the bare minimum required — the smallest set of cases that would catch a real regression. Do not add exhaustive edge cases, happy-path variations, or tests that duplicate coverage.
