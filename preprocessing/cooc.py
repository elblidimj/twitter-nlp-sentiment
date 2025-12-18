#!/usr/bin/env python3
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Configuration
VOCAB_PATH = "vocab.pkl"
OUTPUT_PATH = "cooc.pkl"
FILES = [
    "twitter-datasets/train_pos_full_processed.txt", 
    "twitter-datasets/train_neg_full_processed.txt"
]

def line_generator(files):
    """Reads files line by line to save RAM."""
    for fn in files:
        if os.path.exists(fn):
            print(f"Processing {fn}...")
            with open(fn, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    # We return the line as is; the Vectorizer handles the splitting
                    yield line
        else:
            print(f"Warning: {fn} not found.")

def main():
    print("--- Building Co-occurrence Matrix (Optimized) ---")

    # 1. Load Vocabulary
    if not os.path.exists(VOCAB_PATH):
        print(f"Error: {VOCAB_PATH} not found. Run your vocab script first.")
        return
    
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    print(f"Loaded vocabulary with {len(vocab)} words.")

    # 2. Vectorize using Scikit-Learn (Creates Document-Term Matrix X)
    # binary=True makes the result identical to your 'tokens = [t for t in tokens if t >= 0]' logic
    vectorizer = CountVectorizer(
        vocabulary=vocab,
        tokenizer=lambda x: x.split(), # Matches your original text.split()
        binary=True,                   # Counts word presence (1/0) per tweet
        token_pattern=None             # Required for custom tokenizer
    )

    print("Step 1: Creating sparse Document-Term matrix (X)...")
    # This creates a sparse matrix of size (Total_Tweets, Vocab_Size)
    X = vectorizer.transform(line_generator(FILES))

    # 3. Compute Co-occurrence using Linear Algebra: X^T * X
    # This mathematical trick calculates all co-occurrences in one go
    print(f"Step 2: Computing Co-occurrence (X.T * X) for {X.shape[0]} tweets...")
    cooc = X.T.dot(X)

    # 4. Optional: Set diagonal to zero
    # (Only if you don't want a word to co-occur with itself)
    # cooc.setdiag(0)

    # 5. Save the result
    print(f"Step 3: Saving matrix with {cooc.nnz} non-zero entries...")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)
    
    print(f"✅ Done! Co-occurrence matrix saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()