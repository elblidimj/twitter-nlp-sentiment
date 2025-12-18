from torch.utils.data import Dataset
import numpy as np
from src.utils.io_utils import load_stopwords, load_vocab_and_embeddings
from src.transforms.text_embeddings import tweets_to_cnn


class TweetDataset(Dataset):
    def __init__(self, tweets,y):
        vocab_path="vocab.pkl"
        emb_path="embeddings.npy"
        vocab, embeddings = load_vocab_and_embeddings(vocab_path, emb_path)
        stopwords = load_stopwords("stopwords.pkl")
        self.tweets = tweets_to_cnn(tweets, vocab, embeddings, stopwords)
        y[y == -1] = 0
        self.y = y
        
    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        return self.tweets[idx], self.y[idx]
    
