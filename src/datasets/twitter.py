# src/datasets/twitter.py

from pathlib import Path
import numpy as np
from src.utils.text_analysis import get_word_frequencies, plot_top_words  # cf section suivante

def load_training_tweets(data_dir="twitter-datasets", use_full=True,
                         stopwords=None, do_plots=False):
    data_dir = Path(data_dir)

    if use_full:
        pos_path = data_dir / "train_pos_full_processed.txt"
        neg_path = data_dir / "train_neg_full_processed.txt"
    else:
        pos_path = data_dir / "train_pos_processed.txt"
        neg_path = data_dir / "train_neg_processed.txt"

    with open(pos_path, "r", encoding="utf-8") as f:
        pos_tweets = [line.strip() for line in f]
    with open(neg_path, "r", encoding="utf-8") as f:
        neg_tweets = [line.strip() for line in f]

    if do_plots:
        pos_counter = get_word_frequencies(pos_tweets, min_len=2, stopwords=stopwords)
        neg_counter = get_word_frequencies(neg_tweets, min_len=2, stopwords=stopwords)
        plot_top_words(pos_counter, top_k=20, title="Top 20 Positive Words (stopwords removed)")
        plot_top_words(neg_counter, top_k=20, title="Top 20 Negative Words (stopwords removed)")

    tweets = pos_tweets + neg_tweets
    y_pos = np.ones(len(pos_tweets), dtype=int)
    y_neg = -np.ones(len(neg_tweets), dtype=int)
    y = np.concatenate([y_pos, y_neg])

    return tweets, y


def load_test_tweets(data_dir="twitter-datasets"):
    data_dir = Path(data_dir)
    test_path = data_dir / "test_data_processed.txt"

    with open(test_path, "r", encoding="utf-8") as f:
        tweets = [line.strip() for line in f]

    ids = np.arange(1, len(tweets) + 1)
    return ids, tweets
