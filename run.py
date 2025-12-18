from sklearn.model_selection import train_test_split
from src.model.bertweet import build_model
from src.trainer.bertweet_train import train_bert, predict_bert
from src.datasets.twitter import load_training_tweets, load_test_tweets
from src.trainer.trainer_bilstm import predict_lstm, train_lstm, grid_lstm
from src.utils.io_utils import load_vocab_and_embeddings
from src.model.bilstm import build_lstm
from src.transforms.text_embeddings import tweets_to_matrix
import argparse
import pandas as pd
import torch
import random
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Tweet sentiment classification")

    parser.add_argument(
        "--model",
        type=str,
        default="bertweet",
        choices=["bertweet", "bilstm", "cnn"],
        help="Model to use (default: bertweet)"
    )

    parser.add_argument(
        "--cv",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Enable cross-validation (True / False)"
    )

    return parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    args = parse_args()

    print("Loading data...")
    train_texts, train_labels = load_training_tweets(use_full=False)
    test_ids, test_texts = load_test_tweets()

    X_train, X_val, y_train, y_val = train_test_split(
        train_texts,
        train_labels,
        test_size=0.2,
        random_state=42,
        stratify=train_labels
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, embeddings = load_vocab_and_embeddings("vocab.pkl", "embeddings.npy")

    if args.model == "bertweet":
        from src.model.bertweet import build_model
        from src.trainer.bertweet_train import train_bert, predict_bert

        model = build_model(device)

        train_bert(model,
            X_train, y_train,
            X_val, y_val,
            device=device
        )

        predictions = predict_bert(
            test_texts,
            model=model,
            device=device
        )

    elif args.model == "bilstm":
        hidden_size = 128
        dropout_rate = 0.5
        learning_rate = 0.005
        X_train =tweets_to_matrix(X_train, vocab, embeddings, None)
        X_val = tweets_to_matrix(X_val, vocab, embeddings, None)

        if args.cv is True:
            learning_rate, hidden_size, dropout_rate = grid_lstm(
    X_train, y_train,
    X_val, y_val,
    embeddings,
    device
)


        model = build_lstm(embeddings, hidden_size, dropout_rate)

        train_lstm(X_train, y_train, model, device, embeddings, learning_rate)

        X_test = tweets_to_matrix(test_texts, vocab, embeddings, None)
        predictions = predict_lstm(model, device, X_test)

    elif args.model == "cnn":
        from src.trainer.trainer_cnn import train_and_predict

        train_and_predict(submission_name="submission_cnn.csv")
        exit(0)

    else:
        raise ValueError(f"Unknown model {args.model}")


    pd.DataFrame({
        "Id": test_ids,
        "Prediction": predictions
    }).to_csv("submission.csv", index=False)
