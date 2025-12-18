from sklearn.model_selection import train_test_split
from src.model.bertweet import build_model
from src.trainer.bertweet_train import train_bert, predict_bert
from src.datasets.twitter import load_training_tweets, load_test_tweets
from src.trainer.trainer_bilstm import predict_lstm, train_lstm, grid_lstm
from src.utils.io_utils import load_vocab_and_embeddings
from src.model.bilstm import build_lstm
from src.transforms.text_embeddings import tweets_to_matrix
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

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

def eval_from_predictions(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, f1


if __name__ == '__main__':
    args = parse_args()
    set_seed(42)
    print("Loading data...")
    train_texts, train_labels = load_training_tweets(use_full=True)
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
        val_preds = predict_bert(
        X_val,
        model=model,
        device=device
    )

        val_acc, val_f1 = eval_from_predictions(y_val, val_preds)
        print(f"\n[BERTweet] Val Accuracy: {val_acc:.4f} | Val F1: {val_f1:.4f}")

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
        model.to(device)
        train_lstm(X_train, y_train, model, device, embeddings, learning_rate)

        X_test = tweets_to_matrix(test_texts, vocab, embeddings, None)
        predictions = predict_lstm(model, device, test_ids, X_test,embeddings)
        val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_val).long(),
            torch.from_numpy(y_val).float()
        ),
        batch_size=512,
        shuffle=False
    )
        model.eval()

        val_preds = []
        val_targets = []

        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                by = by.to(device)

                probs = model(bx).squeeze(1)   # shape: (B,)
                preds = (probs > 0.5).long()   # {0,1}

                val_preds.append(preds.cpu())
                val_targets.append(by.cpu())

        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1  = f1_score(val_targets, val_preds)

        print(f"[BiLSTM] Val Accuracy: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    elif args.model == "cnn":
        from src.model.cnn import build_cnn
        from src.trainer.trainer_cnn import train_cnn, predict_cnn, grid_cnn
    
        # Default params
        kernel_size = 3
        filters = 128
        learning_rate = 0.0005
        
        X_train_mat = tweets_to_matrix(X_train, vocab, embeddings, None)
        X_val_mat = tweets_to_matrix(X_val, vocab, embeddings, None)
    
        if args.cv is True:
            learning_rate, kernel_size, filters = grid_cnn(
                X_train_mat, y_train, 
                X_val_mat, y_val, 
                embeddings, device
            )
    
        model = build_cnn(embeddings, kernel_size=kernel_size, filters=filters).to(device)
        train_cnn(X_train_mat, y_train, model, device, embeddings, learning_rate)
    
        X_test = tweets_to_matrix(test_texts, vocab, embeddings, None)
        predictions = predict_cnn(model, device, X_test, embeddings)

        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_val_mat).long(),
                torch.from_numpy(y_val).float()
            ),
            batch_size=512,
            shuffle=False
        )
        model.eval()

        val_preds = []
        val_targets = []

        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                by = by.to(device)

                # CNN output shape check: squeeze if necessary to match (B,)
                probs = model(bx).squeeze()   
                preds = (probs > 0.5).long()   # Thresholding for binary {0,1}

                val_preds.append(preds.cpu())
                val_targets.append(by.cpu())

        # Concatenate and convert to numpy for sklearn metrics
        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()

        # Metrics calculation
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1  = f1_score(val_targets, val_preds)

        print(f"[CNN] Final Val Accuracy: {val_acc:.4f} | Val F1: {val_f1:.4f}")
    else:
        raise ValueError(f"Unknown model {args.model}")


    pd.DataFrame({
        "Id": test_ids,
        "Prediction": predictions
    }).to_csv("submission.csv", index=False)
