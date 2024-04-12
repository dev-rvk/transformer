import tiktoken


class Tokenizer:
    def __init__(self, model_name="cl100k_base"):
        self.encoding = tiktoken.get_encoding(model_name)

    def encode(self, text):
        return self.encoding.encode(text)

    def decode(self, encoded_text):
        return self.encoding.decode(encoded_text)
