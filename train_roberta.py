import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------
# Dataset
# -------------------------
class TwitterSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# -------------------------
# Load data
# -------------------------
def load_train_data(data_dir="twitter-datasets", use_full=True):
    if use_full:
        pos_path = f"{data_dir}/train_pos_full.txt"
        neg_path = f"{data_dir}/train_neg_full.txt"
    else:
        pos_path = f"{data_dir}/train_pos.txt"
        neg_path = f"{data_dir}/train_neg.txt"

    with open(pos_path, "r", encoding="utf-8") as f:
        pos = [line.strip() for line in f]
    with open(neg_path, "r", encoding="utf-8") as f:
        neg = [line.strip() for line in f]

    texts = pos + neg
    # For RoBERTa classification head: labels must be 0/1
    labels = [1] * len(pos) + [0] * len(neg)
    return texts, labels


# -------------------------
# Train
# -------------------------
def train_roberta(
    data_dir="twitter-datasets",
    model_name="roberta-base",
    epochs=1,
    batch_size=16,
    lr=2e-5,
    max_len=128
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    print(">>> Tokenizer and model loaded")
    print(">>> Model on device:", device)

    texts, labels = load_train_data(data_dir=data_dir, use_full=True)

    print(">>> Data loaded")
    print(">>> Number of samples:", len(texts))
    print(">>> First example:", texts[0][:100])
    print(">>> Labels distribution:",
        sum(labels), "/", len(labels))

    # Simple train/val split (90/10)
    idx = np.arange(len(texts))
    np.random.seed(42)
    np.random.shuffle(idx)

    split = int(0.9 * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    print(">>> Train size:", len(train_texts))
    print(">>> Val size:", len(val_texts))

    train_ds = TwitterSentimentDataset(train_texts, train_labels, tokenizer, max_len=max_len)
    val_ds = TwitterSentimentDataset(val_texts, val_labels, tokenizer, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(">>> DataLoaders created")
    print(">>> Train batches:", len(train_loader))
    print(">>> Val batches:", len(val_loader))

    optimizer = AdamW(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch+1}/{epochs} =====")
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            if step % 50 == 0:
                print(f"Epoch {epoch+1} | Step {step}/{len(train_loader)}")

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} finished | Avg loss: {total_loss / len(train_loader):.4f}")


        # Validation: accuracy
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                labels_batch = batch["labels"].numpy()  # ground truth on CPU

                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                correct += (preds == labels_batch).sum()
                total += len(labels_batch)

        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Val accuracy: {acc:.4f}")



if __name__ == "__main__":
    print(">>> Script started")
    train_roberta(epochs=1, batch_size=16, lr=2e-5, max_len=128)
