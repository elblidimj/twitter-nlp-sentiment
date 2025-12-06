# src/trainer/twitter_trainer.py

import numpy as np
from helpers import create_csv_submission
from src.utils.io_utils import load_stopwords, load_vocab_and_embeddings
from src.datasets.twitter import load_training_tweets, load_test_tweets
from src.transforms.text_embeddings import tweets_to_matrix
from src.model.logreg import build_logreg   # ou from model.svm import build_linear_svm

def train_and_predict(
    data_dir="twitter-datasets",
    vocab_path="vocab.pkl",
    emb_path="embeddings.npy",
    submission_name="submission.csv"
):
    vocab, embeddings = load_vocab_and_embeddings(vocab_path, emb_path)
    stopwords = load_stopwords("stopwords.pkl")

    tweets_train, y_train = load_training_tweets(
        data_dir=data_dir, use_full=True, stopwords=stopwords, do_plots=False
    )
    X_train = tweets_to_matrix(tweets_train, vocab, embeddings, stopwords)

    classifier = build_logreg(C=1.0, max_iter=1000)
    classifier.fit(X_train, y_train)

    test_ids, test_tweets = load_test_tweets(data_dir=data_dir)
    X_test = tweets_to_matrix(test_tweets, vocab, embeddings, stopwords)
    y_test_pred = classifier.predict(X_test)
    y_test_pred = np.where(y_test_pred >= 0, 1, -1).astype(int)

    create_csv_submission(test_ids, y_test_pred, submission_name)
    print(f"Submission file created: {submission_name}")
