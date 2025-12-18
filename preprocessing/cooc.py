#!/usr/bin/env python3
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

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
                    yield line
        else:
            print(f"Warning: {fn} not found.")

def main():
    print("--- Building Co-occurrence Matrix (Optimized) ---")

    if not os.path.exists(VOCAB_PATH):
        print(f"Error: {VOCAB_PATH} not found. Run your vocab script first.")
        return
    
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    print(f"Loaded vocabulary with {len(vocab)} words.")

    vectorizer = CountVectorizer(
        vocabulary=vocab,
        tokenizer=lambda x: x.split(),
        binary=True,                   
        token_pattern=None             
    )

    print("Step 1: Creating sparse Document-Term matrix (X)...")
    X = vectorizer.transform(line_generator(FILES))

    print(f"Step 2: Computing Co-occurrence (X.T * X) for {X.shape[0]} tweets...")
    cooc = X.T.dot(X)

    print(f"Step 3: Saving matrix with {cooc.nnz} non-zero entries...")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)
    
    print(f"Co-occurrence matrix saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()