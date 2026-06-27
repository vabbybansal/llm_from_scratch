import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_from_scratch.libs.utils import get_device

def get_tokenizer(model_name="meta-llama/Llama-3.2-1B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # base models ship without a chat template; borrow the official Llama 3 template from the Instruct variant
    if tokenizer.chat_template is None:
        instruct = AutoTokenizer.from_pretrained(model_name + "-Instruct")
        tokenizer.chat_template = instruct.chat_template
    return tokenizer

def get_model(model_name="meta-llama/Llama-3.2-1B"):
    tokenizer = get_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=get_device()
    )
    return tokenizer, model

def generate(s: str, tokenizer, model):
    # chat-format the prompt so a post-SFT model triggers its learned instruction-following behavior.
    # add_generation_prompt=True appends the assistant header so the model knows to start responding.
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": s}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(get_device())
    # stop on the turn-ender <|eot_id|> (what SFT teaches the model to emit) as well as the base eos;
    # otherwise generate() runs past the response end and loops until max_new_tokens
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.8, top_k=50,
                             eos_token_id=[tokenizer.eos_token_id, eot_id],
                             pad_token_id=tokenizer.eos_token_id)  # set explicitly to silence the HF default-pad warning
    # decode only the newly generated tokens (skip the prompt) for a clean response
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


if __name__ == '__main__':
    t, m = get_model()

    for prompt in [
        "What is the capital of France?",
        "Write a haiku about the ocean.",
        "Explain the difference between machine learning and deep learning.",
        "Give me three tips for staying productive.",
        "Summarize what a transformer model is in one sentence."
    ]:
        print("****************************************")
        print(prompt)
        print(generate(prompt, t, m))