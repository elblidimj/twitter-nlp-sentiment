import numpy as np
import pickle

def load_pretrained_glove(vocab, glove_path="glove.twitter.27B.25d.txt"):
    """
    Creates an embedding matrix for your specific vocab using pre-trained GloVe vectors.
    """
    print(f"Loading GloVe vectors from {glove_path}...")

    # 1. Read the GloVe file to get dimensions
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            # Verify the word is in our vocab to save RAM (optional but recommended)
            if word in vocab:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

    # Determine embedding dimension from the first found vector
    # (GloVe Twitter comes in 25d, 50d, 100d, 200d)
    if embeddings_index:
        emb_dim = len(next(iter(embeddings_index.values())))
    else:
        emb_dim = 25 # Default fallback

    print(f"Found {len(embeddings_index)} words in GloVe (Dim: {emb_dim})")

    # 2. Create the Embedding Matrix
    # Initialize with random noise or zeros
    vocab_size = len(vocab)
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, emb_dim)).astype(np.float32)
    # Or use zeros: embedding_matrix = np.zeros((vocab_size, emb_dim))

    hits = 0
    misses = 0

    # 3. Map Vocab words to GloVe vectors
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1

    print(f"Converted {hits} words ({hits/vocab_size:.1%}) found in GloVe.")
    print(f"Missed {misses} words (randomly initialized).")

    return embedding_matrix

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
embeddings = load_pretrained_glove(vocab, glove_path="glove.twitter.27B.50d.txt")
np.save("embeddings.npy", embeddings)