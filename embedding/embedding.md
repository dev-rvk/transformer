# Embedding

We have mapped the works to a specific id in the vocabulary using tokenization. 

Now we need to convert it into an embedding vector of size 512 which is the dimensionality of our model `d_model`.

Within embedding there are two layers:
- Input Embedding
- Positional embedding

## Input Embedding
This layer is used to map every integer in the vocabulary (within range of vocabulary) to a specific vector of size `d_model` i.e. 512.

For Tokenization, we have used cl100k_base using tiktoken which has a vocabulary size of `100256`, so `vocab_size = 100256`.

Note: the embedding vector must be the same for a particular vocabulary index.

We are using Embedding from pytorch for generating the embedding vector.

## Positional Embedding

We create some special positional encoding vectors that are added to the input embedding vectors of each token.

Note: These vectors are created only once and are reused for every sentence
![positional_encoding.png](..%2Fimages%2Fpositional_encoding.png)
- Create a matrix `pe` of shape (seq_len, d_model) initialized to 0
- Create a vector `position` of shape (seq_len)
- Create a vector `div_term` of shape (d_model) 
- Apply sine to even indices
- Apply cosine to odd indices
- Add a batch dimension to the positional encoding
- Register the positional encoding as a buffer

`self.dropout:` This line initializes a dropout layer with a dropout probability of 0.1. Dropout is a regularization technique used to prevent overfitting by randomly setting a fraction of input units to zero during training.
`requires_grad_(False):` This line sets the requires_grad attribute of the positional encodings (pe) to False. This is done because positional encodings are not trainable parameters; they are fixed and do not need gradients calculated during backpropagation.


