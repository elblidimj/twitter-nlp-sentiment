import numpy as np

MAX_LEN = 30 

def tweet_to_vec_indices(tweet, vocab, stopwords=None, max_len=MAX_LEN):
    """
    Convert a tweet into a vector of indices
    """
    tokens = tweet.strip().split()
    indices = []

    for w in tokens:
        if stopwords and w in stopwords:
            continue
        idx = vocab.get(w, 0)
        if idx > 0: 
            indices.append(idx)

    if len(indices) > max_len:
        indices = indices[:max_len]

    padded_sequence = np.zeros(max_len, dtype=np.int32)
    padded_sequence[:len(indices)] = indices
    
    return padded_sequence

def tweets_to_matrix(tweets, vocab, embeddings, stopwords=None):
    X = np.zeros((len(tweets), MAX_LEN), dtype=np.int32) 
    
    for i, t in enumerate(tweets):
        X[i] = tweet_to_vec_indices(t, vocab, stopwords)
        
    return X