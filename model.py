from tokenizer.tokenizer import Tokenizer
from embedding.embedding import InputEmbedding, AbsolutePositionalEncoder

VOCAB_SIZE = 100256
D_MODEL = 512

def main():
    # take input
    input_text = "hello world!"

    # Define Blocks
    tokenizer = Tokenizer()
    input_encoder = InputEmbedding(vocab_size=VOCAB_SIZE, d_model=D_MODEL)
    positional_encoder = AbsolutePositionalEncoder(emb_dim=D_MODEL)

    tokens = tokenizer.encode(input_text)

    input_embedding = input_encoder(tokens)
    positional_encoder = positional_encoder(input_embedding)

    # Decode
    # decoded = tokenizer.decode(encoded)
    print()


if __name__ == "__main__":
    main()
