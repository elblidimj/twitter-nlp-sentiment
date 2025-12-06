import pickle
import numpy as np

def load_stopwords(path="stopwords.pkl"):
    with open(path, "rb") as f:
        stopwords = pickle.load(f)
    # add <user> as extra stopword
    stopwords = set(stopwords)
    stopwords.add("<user>")
    return stopwords

def load_vocab_and_embeddings(vocab_path="vocab.pkl", emb_path="embeddings.npy"):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    embeddings = np.load(emb_path)
    return vocab, embeddings
