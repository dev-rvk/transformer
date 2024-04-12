# Tokenizer
Tokenizers transform human language into a format that GPT models can understand and work with efficiently
The tokenizer is used to convert then input string to integers (encode) or convert integers back to strings.
This project uses BPE Tokenizer.

The tokenizer creates a vocabulary. Long size of vocabulary means small list of integers is generated and vice versa.

### Byte Pair Encoding Tokenizer
1. **Pre-processing**: The BPE tokenizer starts by segmenting the text into its basic units. This can involve splitting the text into individual characters or words depending on the specific implementation.
2. **Learning Merges**: It then analyzes the characters or words (depending on the pre-processing step) and identifies the most frequently occurring pairs. These pairs can be individual characters next to each other (like "th" or "ing") or whole words that frequently appear together (like "of the").
3. **Merging and Vocabulary Building**: The most frequent pair is then merged into a single new token. This essentially creates a new word in the tokenizer's vocabulary.
The tokenizer then re-analyzes the text, considering the newly formed token along with the existing characters/words.
4. **Iteration and Vocabulary Growth**: Steps 2 and 3 are repeated iteratively. In each iteration, the tokenizer finds the most frequent pair (considering both existing characters/words and previously merged tokens) and merges them into a new token.
With each merge, the vocabulary of the tokenizer keeps growing, incorporating both individual characters/words and the newly created merged tokens.
5. **Stopping Criteria**: This iterative merging process continues until a predefined vocabulary size is reached, or some other stopping criterion is met. This stopping point controls the trade-off between handling rare words and model efficiency.

### OpenAI's tiktoken
tiktoken is a fast BPE tokeniser for use with OpenAI's models.


