import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_from_scratch.libs.utils import get_device

def get_model(model_name="meta-llama/Llama-3.2-1B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        dtype=torch.bfloat16,
        device_map=get_device()
    )
    return tokenizer, model

def generate(s: str, tokenizer, model):
    inputs = tokenizer(s, return_tensors="pt").to(get_device())
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.8, top_k=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == '__main__':
    t, m = get_model()

    for prompt in [
        "The earth is flat because",
        "King minus male plus female equals",
        "The capital of France is Paris, and the capital of Germany is",
        "Once upon a time in a land far away, there lived a",
        "The difference between machine learning and deep learning is"
    ]:
        print("****************************************")
        print(prompt)
        print(generate(prompt, t, m))