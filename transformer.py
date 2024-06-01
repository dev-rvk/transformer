import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout)
        self.decoder = Decoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout)

    def forward(self, src_tokens, trg_tokens):
        # Create padding mask for the source tokens
        src_padding_mask = create_padding_mask(src_tokens)

        # Encode source tokens
        encoder_output = self.encoder(src_tokens, src_padding_mask)

        # Create self-attention mask for the decoder
        trg_self_mask = create_padding_mask(trg_tokens)

        # Decode using encoder output
        decoder_output = self.decoder(trg_tokens, encoder_output, trg_self_mask, src_padding_mask)

        return decoder_output

    def generate(self, src_tokens, max_length=100, start_token=1, end_token=2):
        """
        Generate text based on the source tokens.

        Args:
        - src_tokens (torch.Tensor): Source tokens tensor of shape (seq_len,).
        - max_length (int): Maximum length of the generated sequence.
        - start_token (int): Start token ID.
        - end_token (int): End token ID.

        Returns:
        - generated_tokens (list): List of generated token IDs.
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # Encode source tokens
            encoder_output = self.encoder(src_tokens.unsqueeze(0), None)

            # Initialize target tokens with start token
            trg_tokens = [start_token]

            for _ in range(max_length):
                # Convert trg_tokens to tensor
                trg_tensor = torch.tensor(trg_tokens).unsqueeze(0)

                # Create self-attention mask for the decoder
                trg_self_mask = create_padding_mask(trg_tensor)

                # Decode using encoder output
                decoder_output = self.decoder(trg_tensor, encoder_output, trg_self_mask, None)

                # Get the last predicted token (highest probability)
                next_token = torch.argmax(decoder_output[:, -1, :], dim=-1).item()

                # Append next_token to trg_tokens
                trg_tokens.append(next_token)

                # Break if end_token is generated
                if next_token == end_token:
                    break

        return trg_tokens


VOCAB_SIZE = 100256
D_MODEL = 512
NUM_HEADS = 8
D_FF = 2048

# Initialize Transformer model
transformer_model = Transformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    num_layers=5,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    max_seq_length=5000
)

# Example tokens for source and target
src_tokens = torch.tensor([15339, 0, 32])  # Example source tokens
trg_tokens = torch.tensor([0, 32, 64])  # Example target tokens

# Get decoder output
decoder_output = transformer_model(src_tokens.unsqueeze(0), trg_tokens.unsqueeze(0))

print(decoder_output.shape)
