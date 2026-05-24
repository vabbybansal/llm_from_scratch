import tiktoken

class Tokenizer:
    def __init__(self, model_name: str):
        self.tokenizer = tiktoken.get_encoding(model_name)
    
    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)