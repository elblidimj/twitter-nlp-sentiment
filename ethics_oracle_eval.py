import torch
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.datasets.twitter import load_training_tweets


def main():
    """
    This script evaluates the alignment between emoji-based polarity labels and
    a strong external sentiment analysis model (oracle).

    The goal is not to improve classification performance, but to quantify how
    often a sentiment classifier agrees with emoji polarity labels, in order to
    assess the risk of misinterpreting emoji polarity as true sentiment or emotion.
    """

    # Load the training tweets and their emoji-based polarity labels.
    # Labels are converted to a binary format: 1 for positive, 0 for negative.
    texts, y = load_training_tweets(use_full=False)
    y_bin = np.array([1 if int(v) == 1 else 0 for v in y])

    # Load a pre-trained sentiment analysis model as an oracle.
    # This model was trained independently on Twitter data and serves as a
    # strong reference for sentiment prediction.
    oracle_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(oracle_name)
    model = AutoModelForSequenceClassification.from_pretrained(oracle_name)

    # Run inference on GPU if available for efficiency.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    preds = []
    batch_size = 64

    # Perform batched inference without gradient computation.
    # Tweets are tokenized, forwarded through the model, and converted to
    # binary predictions for comparison with emoji polarity labels.
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(texts)} tweets")

            batch = texts[i:i + batch_size]
            encodings = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            )
            encodings = {k: v.to(device) for k, v in encodings.items()}

            logits = model(**encodings).logits

            # The oracle model predicts three classes (negative, neutral, positive).
            # To obtain binary predictions, neutral cases are resolved by comparing
            # positive and negative logits, effectively discarding neutrality.
            if logits.shape[1] == 3:
                pred = (logits[:, 2] > logits[:, 0]).long()
            else:
                pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().numpy().tolist())

    # Compute standard classification metrics to quantify the agreement
    # between oracle sentiment predictions and emoji-based polarity labels.
    acc = accuracy_score(y_bin, preds)
    f1 = f1_score(y_bin, preds)
    cm = confusion_matrix(y_bin, preds)

    print(f"Oracle sentiment vs emoji-polarity | Acc={acc:.4f} F1={f1:.4f}")
    print("Confusion matrix:\n", cm)


if __name__ == "__main__":
    main()
