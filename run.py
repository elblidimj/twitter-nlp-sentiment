from sklearn.model_selection import train_test_split
from src.model.bertweet import build_model
from src.trainer.bertweet_train import train_bert, predict_bert
from src.datasets.twitter import load_training_tweets, load_test_tweets
import argparse
import pandas as pd
import torch

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


    if args.model == "bertweet":
        from src.model.bertweet import build_model
        from src.trainer.bertweet_train import train_bert, predict_bert

        model = build_model(device)

        train_bert(
            X_train, y_train,
            X_val, y_val,
            model=model,
            device=device
        )

        predictions = predict_bert(
            test_texts,
            model=model,
            device=device
        )

    elif args.model == "bilstm":
        from src.trainer.trainer_bilstm import train_and_predict_bilstm

        if args.cv is False:
            print("⚠️ BiLSTM currently ALWAYS uses CV (hardcoded). Ignoring --cv flag.")

        train_and_predict_bilstm()
        exit(0)

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
