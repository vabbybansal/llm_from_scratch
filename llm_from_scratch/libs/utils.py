import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM


# TODO (cross-stage / future): this lives in shared libs/ but is currently GPT-specific —
#   it assumes model(x) returns a raw logits tensor and takes an explicit context_size.
#   Plan: extend it to be model-agnostic (normalize HF `.logits`, accept context_size) so SFT/DPO
#   can share one inference path, then deprecate the HF-native generate in load_pretrained_hf_model.py.
# TODO (not production-ready — gaps vs lab inference):
# 1. No KV cache — recomputes attention for all tokens every step (O(n^2) cost); labs cache past K/V
# 2. No padding mask — finished sequences in a batch keep running wasted compute instead of being masked out
# 3. Repetition penalty applied via Python loop — not vectorized, slow for large batches
# 4. EOS stops entire batch — should mask finished sequences and continue for unfinished ones
# 5. No beam search — only greedy/sampling; labs support beam search for higher quality outputs
# 6. No min_new_tokens — can't prevent EOS from firing too early
# 7. No logits processors / warpers pipeline — labs have a composable filter chain (bad words, no repeat ngram, etc.)
# 8. Inference optimization: quantization (int8/int4), speculative decoding, continuous batching (vLLM/TGI)
def generate(model, input_ids, max_new_tokens, context_size, temperature=1.0, top_k=None, top_p=None, eos_token_id=None, repetition_penalty=1.0):

    # input_ids  # (b, prompt_input_tokens)
    for _ in range(max_new_tokens):
        # input_ids can contain many more tokens than the context size. For predictions, get the context size worth of tokens from the last since the model can only handle that
        input_cond = input_ids[:, -context_size:]  # (b, context_size)
        with torch.no_grad():
            logits = model(input_cond)  # (b, context_size, vocab_size)

        # logits[:, -1, :]  # (b,vocab_size) => all logits for the vocab_size neurons at the last token index. Middle index collapses automatically since we reference a single index
        logits_last_token = logits[:, -1, :]
        if repetition_penalty != 1.0:
            for b_idx in range(input_ids.shape[0]):
                for token_id in input_ids[b_idx].tolist():
                    logits_last_token[b_idx, token_id] /= repetition_penalty

        if temperature == 0:  # apply greedy
            # torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # chooses the token index with max value across the batch. Collapses to shape (b) unless keepdim is passed as true which makes it (b,1)
            token_next = torch.argmax(logits_last_token, dim=-1, keepdim=True)  # (b, 1)
        else:   # sample from prob distribution
            logits_last_token = logits_last_token / temperature

            if top_k:
                top_k_vals = torch.topk(logits_last_token, top_k, dim=-1).values
                top_k_vals_smallest = top_k_vals[...,-1,None]  # ... makes the shape invariant. -1 selects the last tensor in each batch. None adds a dimension in the end for broadcasting
                logits_last_token[logits_last_token < top_k_vals_smallest] = float('-inf')  # inplace modification
            
            if top_p:
                sorted_logits_last_token, sorted_idx = torch.sort(logits_last_token, descending=True, dim=-1)  # sort and store values and indices
                cum_probs = torch.cumsum(torch.softmax(sorted_logits_last_token, dim=-1), dim=-1)  # create prob and do cumulative sum
                # remove tokens once cumulative prob exceeds top_p
                sorted_logits_last_token[cum_probs - torch.softmax(sorted_logits_last_token, dim=-1) >= top_p] = float('-inf')
                logits_last_token = sorted_logits_last_token.scatter(-1, sorted_idx, sorted_logits_last_token)  # -1 is the dimension. Scatters back the data to the right places using vals and indices
            
            # torch.softmax(logits[:, -1, :], dim=-1, keepdim=True)  # same shape. Converts to a prob dist
            dist = torch.softmax(logits_last_token, dim=-1)  # creates a prob distribution
            token_next = torch.multinomial(dist, num_samples=1)  # samples from the prob distribution which is a multinomial distribution => (b, 1)
            
        input_ids = torch.cat((input_ids, token_next), dim=-1)  # torch.cat((b,x), (b,1)) with dim==-1 => (b,x+1)
        if eos_token_id is not None and (token_next == eos_token_id).all():
            break
    return input_ids


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def create_rm_classifier_from_lm_hf(base_model_name_hf, sft_checkpoint_path, sft_model_dtype, tokenizer) -> AutoModelForSequenceClassification:

    # Use HF to automatically put a classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name_hf,
        num_labels=1,
        torch_dtype=sft_model_dtype
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # warm the weights from the sft trained model
    sd = torch.load(sft_checkpoint_path)["model_state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"SFT Model Loaded into RM classification head. New additions - {missing}, dropped - {unexpected}")
    return model

    
def create_rm_classifier_from_lm_scratch(base_model_name_hf, sft_checkpoint_path, sft_model_dtype):
    
    # AutoModelForCausalLM has two composable parts: one is the core backbone(.model) and the other is the disposable vocab projection (.lm_head)
    # We want to scrap the vocab projection, and attach a linear on top of the backbone which has most of the LM knowledge
    causal_lm = AutoModelForCausalLM.from_pretrained(
        base_model_name_hf,
        torch_dtype=sft_model_dtype
    )

    sd = torch.load(sft_checkpoint_path)["model_state_dict"]
    missing, unexpected = causal_lm.load_state_dict(sd, strict=False)

    class RewardModelFromHF(nn.Module):
        def __init__(self, causal_lm):
            super().__init__()
            self.backbone = causal_lm.base_model
            d = causal_lm.config.hidden_size
            self.score = nn.Linear(d, 1, bias=False)
            # match the backbone's dtype
            self.score = self.score.to(next(self.backbone.parameters()).dtype)
    
        def forward(self, input_ids, attention_mask):
            wrap_x = self.backbone(input_ids=input_ids, attention_mask=attention_mask)  
            x = wrap_x.last_hidden_state  # (b, context_size, d)
            x = self.score(x).squeeze(-1)   # (b, context_size, 1) => (b, context_size)

            last_idx = attention_mask.sum(1) - 1  # last real token. Note that this is only valid if we are doing only right padding. Any left padding breaks it
            b = x.size(0)
            
            # pick the last tensor
            return x[torch.arange(b, device=x.device), last_idx]

    return RewardModelFromHF(causal_lm)
    