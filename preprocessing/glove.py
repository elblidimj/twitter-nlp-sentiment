import numpy as np
import scipy.sparse as sp
import pickle
import os

def train_hybrid_embeddings(
    vocab_path="vocab.pkl",
    cooc_path="cooc.pkl",
    glove_path="glove.twitter.27B.50d.txt",
    output_path="embeddings.npy", # Changed to embeddings.npy to match your trainer
    epochs=15, # 15-20 is usually enough for hybrid
    batch_size=20000
):
    print("Initializing Hybrid Training...")

    # 1. Load Vocab
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    # 2. Load Pre-trained GloVe with Length Validation
    print(f"Loading pre-trained vectors from {glove_path}...")
    glove_map = {}
    emb_dim = 50 # Target dimension

    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                if len(values) != emb_dim + 1:
                    continue # SKIP corrupted lines or lines with wrong dimensions
                
                word = values[0]
                if word in vocab:
                    glove_map[word] = np.asarray(values[1:], dtype='float32')
    except FileNotFoundError:
        print(f"Error: GloVe file {glove_path} not found.")
        return

    print(f"Found {len(glove_map)} overlapping words. Dimension: {emb_dim}")

    # 3. Initialize Vectors (Following your reasoning)
    # Using random normal as in your original script
    xs = np.random.normal(scale=0.1, size=(vocab_size, emb_dim)).astype(np.float32)
    ys = np.random.normal(scale=0.1, size=(vocab_size, emb_dim)).astype(np.float32)

    is_fixed = np.zeros(vocab_size, dtype=bool)

    hits = 0
    for word, idx in vocab.items():
        if word in glove_map:
            vector = glove_map[word]
            # Initialize with half the vector in each matrix
            xs[idx] = vector * 0.5
            ys[idx] = vector * 0.5
            is_fixed[idx] = True
            hits += 1

    print(f"Fixed {hits} words ({hits/vocab_size:.1%}). Training remaining {vocab_size - hits}...")

    # 4. Load Co-occurrence
    if not os.path.exists(cooc_path):
        print(f"Error: {cooc_path} not found.")
        return

    with open(cooc_path, "rb") as f:
        cooc = pickle.load(f)

    if not sp.isspmatrix_coo(cooc):
        cooc = cooc.tocoo()

    nmax = 100
    alpha = 0.75
    eta = 0.001

    weights = np.minimum(1.0, (cooc.data / nmax) ** alpha).astype(np.float32)
    log_coocs = np.log(cooc.data).astype(np.float32)
    indices = np.arange(cooc.nnz)

    # 5. Training Loop
    print("Starting training (updating only non-GloVe words)...")
    for epoch in range(epochs):
        np.random.shuffle(indices)

        for start in range(0, cooc.nnz, batch_size):
            end = min(start + batch_size, cooc.nnz)
            batch_idx = indices[start:end]

            b_rows = cooc.row[batch_idx]
            b_cols = cooc.col[batch_idx]
            b_weights = weights[batch_idx]
            b_log = log_coocs[batch_idx]

            x_vecs = xs[b_rows]
            y_vecs = ys[b_cols]

            preds = np.sum(x_vecs * y_vecs, axis=1)

            grad_factor = 2 * eta * b_weights * (preds - b_log)
            grad_x = grad_factor[:, None] * y_vecs
            grad_y = grad_factor[:, None] * x_vecs

            # Apply gradient masking
            grad_x[is_fixed[b_rows]] = 0
            grad_y[is_fixed[b_cols]] = 0

            np.add.at(xs, b_rows, -grad_x)
            np.add.at(ys, b_cols, -grad_y)

        print(f"Epoch {epoch+1}/{epochs} complete.", end='\r')

    print("\nHybrid training complete.")
    final_embeddings = xs + ys
    np.save(output_path, final_embeddings)
    print(f"Saved hybrid embeddings to {output_path}")
    return final_embeddings

if __name__ == "__main__":
    train_hybrid_embeddings(glove_path="glove.twitter.27B.50d.txt")