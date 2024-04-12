from tokenizers import ByteLevelBPETokenizer

# Initialize the tokenizer
tokenizer = ByteLevelBPETokenizer()

# Train the tokenizer on your text data
corpus_files = ["path/to/corpus.txt"]
tokenizer.train(files=corpus_files, vocab_size=10000, min_frequency=2)

# Save the vocabulary and merges
tokenizer.save("path/to/save/directory")