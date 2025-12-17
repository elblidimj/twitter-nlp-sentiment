import pickle
import numpy as np

def load_stopwords(path="stopwords.pkl"):
    with open(path, "rb") as f:
        stopwords = pickle.load(f)
    stopwords = set(stopwords)
    stopwords.add("<user>")
    stopwords.add("<url>")
    return stopwords

def load_vocab_and_embeddings(vocab_path="vocab.pkl", emb_path="embeddings.npy"):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    embeddings = np.load(emb_path)
    return vocab, embeddings

def load_idf_weights(path="idf_weights.pkl"):    
    try:
        with open(path, "rb") as f:
            idf_weights = pickle.load(f)
        return idf_weights
    except Exception as e:
        print(f"Error loading IDF weights: {e}.")
        return None
