# src/transforms/text_embeddings.py

import numpy as np

def tweet_to_vec(tweet, vocab, embeddings, stopwords=None):
    words = tweet.strip().split()
    vecs = []

    for w in words:
        if stopwords is not None and w in stopwords:
            continue
        idx = vocab.get(w)
        if idx is not None:
            vecs.append(embeddings[idx])

    if not vecs:
        return np.zeros(embeddings.shape[1], dtype=np.float32)

    return np.mean(np.stack(vecs), axis=0).astype(np.float32)


def tweets_to_matrix(tweets, vocab, embeddings, stopwords=None):
    X = np.zeros((len(tweets), embeddings.shape[1]), dtype=np.float32)
    for i, t in enumerate(tweets):
        X[i] = tweet_to_vec(t, vocab, embeddings, stopwords)
    return X
