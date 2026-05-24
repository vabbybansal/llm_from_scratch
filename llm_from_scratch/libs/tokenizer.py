import tiktoken

class Tokenizer:
    GPT_EOS = "<|endoftext|>"  # GPT-family convention

    def __init__(self, model_name: str):
        self.tokenizer = tiktoken.get_encoding(model_name)
    
    def get_eos_string(self) -> str:
        return self.GPT_EOS

    def get_eos_token_id(self) -> int:
        return self.tokenizer.encode_single_token(self.get_eos_string())

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, allowed_special=set([self.get_eos_string()]))

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)