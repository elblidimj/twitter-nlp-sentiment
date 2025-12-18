import torch
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.datasets.twitter import load_training_tweets


def main():
    """
    This script extends the oracle-based evaluation by explicitly handling
    neutral predictions produced by the sentiment model.

    Its purpose is to quantify how often a strong sentiment oracle predicts
    neutral sentiment on tweets that are labeled with binary emoji polarity.
    Neutral cases highlight situations where emoji polarity does not correspond
    to a clear sentiment signal, revealing a fundamental ambiguity in the task.
    """

    # Load the training tweets and their emoji-based polarity labels.
    # Labels are converted to a binary format: 1 for positive, 0 for negative.
    texts, y = load_training_tweets(use_full=False)
    y_bin = np.array([1 if int(v) == 1 else 0 for v in y])

    # Load a pre-trained Twitter-specific sentiment model as an oracle.
    # This oracle predicts three classes: negative, neutral, and positive.
    oracle_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(oracle_name)
    model = AutoModelForSequenceClassification.from_pretrained(oracle_name)

    # Use GPU acceleration when available and switch the model to evaluation mode.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    preds = []
    neutral_flags = []
    batch_size = 64

    # Run batched inference without gradient computation.
    # For each tweet, we record whether the oracle predicts a neutral sentiment.
    # Neutral predictions are treated as intrinsically ambiguous and are not
    # interpreted as meaningful positive or negative sentiment.
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

            logits = model(**encodings).logits  # shape: (batch_size, 3)
            pred_class = torch.argmax(logits, dim=1)  # 0=neg, 1=neu, 2=pos

            for j in range(len(pred_class)):
                if pred_class[j].item() == 1:
                    # Neutral case: no exploitable sentiment signal.
                    # Any binary polarity decision would be arbitrary in this case.
                    neutral_flags.append(True)

                    # A binary label is still assigned for metric computation,
                    # but neutral cases are analyzed separately and not interpreted
                    # as reliable sentiment predictions.
                    preds.append(0)
                else:
                    neutral_flags.append(False)
                    preds.append(1 if pred_class[j].item() == 2 else 0)

    # Compute statistics related to neutral predictions.
    # The neutral ratio quantifies the proportion of tweets for which the
    # sentiment oracle cannot express a clear polarity.
    neutral_flags = np.array(neutral_flags)
    n_neutral = neutral_flags.sum()
    total = len(texts)
    neutral_ratio = n_neutral / total

    # Compute standard agreement metrics between emoji polarity labels
    # and oracle-derived binary sentiment predictions.
    acc = accuracy_score(y_bin, preds)
    f1 = f1_score(y_bin, preds)
    cm = confusion_matrix(y_bin, preds)

    print("\n=== Oracle sentiment vs emoji-polarity ===")
    print(f"Total tweets      : {total}")
    print(f"Neutral (oracle)  : {n_neutral} ({neutral_ratio:.2%})")
    print(f"Accuracy          : {acc:.4f}")
    print(f"F1-score          : {f1:.4f}")
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()
