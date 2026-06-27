from transformers import GPT2Model
from llm_from_scratch.pretraining.model.gpt import GPT
from llm_from_scratch.pretraining.model.constants import GPT2_SMALL


def load_gpt2_weights(model: GPT, hf_model_name: str = "gpt2") -> GPT:
    hf_sd = GPT2Model.from_pretrained(hf_model_name).state_dict()

    # Embeddings — shape matches directly, no transpose
    model.tok_emb.weight.data.copy_(hf_sd["wte.weight"])
    model.pos_emb.weight.data.copy_(hf_sd["wpe.weight"])

    # Final layernorm
    model.ln_f.weight.data.copy_(hf_sd["ln_f.weight"])
    model.ln_f.bias.data.copy_(hf_sd["ln_f.bias"])

    for i, block in enumerate(model.trf_blocks):
        p = f"h.{i}"  # HF prefix for layer i

        # Layer norms — no transpose
        block.ln_1.weight.data.copy_(hf_sd[f"{p}.ln_1.weight"])
        block.ln_1.bias.data.copy_(hf_sd[f"{p}.ln_1.bias"])
        block.ln_2.weight.data.copy_(hf_sd[f"{p}.ln_2.weight"])
        block.ln_2.bias.data.copy_(hf_sd[f"{p}.ln_2.bias"])

        # Attention — HF uses Conv1D: weight shape is (in, out), nn.Linear is (out, in) => .T
        block.attn.c_attn.weight.data.copy_(hf_sd[f"{p}.attn.c_attn.weight"].T)
        block.attn.c_attn.bias.data.copy_(hf_sd[f"{p}.attn.c_attn.bias"])
        block.attn.c_proj.weight.data.copy_(hf_sd[f"{p}.attn.c_proj.weight"].T)
        block.attn.c_proj.bias.data.copy_(hf_sd[f"{p}.attn.c_proj.bias"])

        # MLP — same Conv1D transpose issue
        block.mlp.c_fc.weight.data.copy_(hf_sd[f"{p}.mlp.c_fc.weight"].T)
        block.mlp.c_fc.bias.data.copy_(hf_sd[f"{p}.mlp.c_fc.bias"])
        block.mlp.c_proj.weight.data.copy_(hf_sd[f"{p}.mlp.c_proj.weight"].T)
        block.mlp.c_proj.bias.data.copy_(hf_sd[f"{p}.mlp.c_proj.bias"])

    # lm_head is weight-tied to tok_emb — already covered above
    return model

