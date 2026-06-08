import torch


def generate_text(model, input_ids, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        input_cond = input_ids[:, -context_size:]
        with torch.no_grad():
            logits = model(input_cond)
        token_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat((input_ids, token_next), dim=-1)
    return input_ids


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device