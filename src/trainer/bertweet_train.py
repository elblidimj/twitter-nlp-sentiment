from torch.optim import AdamW

import torch
from src.datasets.bertweet_loader import BERTweetNormalizer
from src.model.bertweet import build_model
from torch.utils.data import DataLoader
from transformers import (
    get_cosine_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score

def fast_evaluate(self, data_loader,device,model):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels']

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            preds = torch.argmax(outputs.logits, dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    return acc, f1

def train(self, X_train, y_train, X_val, y_val,batch_size = 32,epochs=3,device=torch.device('cpu'),lr=0.01):
    model = build_model(device)
    print("\n" + "="*70)
    print("Training")
    print("="*70)
    CSV_FILE = 'log_train.csv'
    GRADIENT_ACCUMULATION = 2
    WARMUP_RATIO = 0.06
    WEIGHT_DECAY = 0.01
    with open(CSV_FILE, 'w') as f:
        f.write("epoch,step,global_step,loss,train_acc,lr,val_acc,val_f1\n")

    print("\n[1/5] Normalizing data...")
    X_train = BERTweetNormalizer.normalize_batch(X_train)
    X_val = BERTweetNormalizer.normalize_batch(X_val)

    train_loader = DataLoader(X_train, y_train, batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(X_val, y_val,batch_size=batch_size, shuffle=False)

    print("\n[3/5] Setting up optimizer...")
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': WEIGHT_DECAY},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)

    total_steps = len(train_loader) * epoch // GRADIENT_ACCUMULATION
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print("\n[4/5] Training...")
    best_val_acc = 0
    global_step = 0

    for epoch in range(epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*70}")

        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc="Training", ncols=100)

        for step, batch in enumerate(progress_bar):

            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / GRADIENT_ACCUMULATION

            loss.backward()

            loss_val = loss.item() * GRADIENT_ACCUMULATION
            train_loss += loss_val

            preds = torch.argmax(outputs.logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            if (step + 1) % GRADIENT_ACCUMULATION == 0:
                optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if step % 10 == 0:
                avg_loss = train_loss / (step + 1)
                current_acc = train_correct / train_total if train_total > 0 else 0
                current_lr = scheduler.get_last_lr()[0]

                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{current_acc:.4f}'})

                with open(CSV_FILE, 'a') as f:
                    f.write(f"{epoch+1},{step},{global_step},{avg_loss:.5f},{current_acc:.5f},{current_lr:.8f},,\n")

        print("Validating...")
        val_acc, val_f1 = fast_evaluate(val_loader,device,model)
        print(f"✓ Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        with open(CSV_FILE, 'a') as f:
            f.write(f"{epoch+1},END,{global_step},,,,{val_acc:.5f},{val_f1:.5f}\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')

    return best_val_acc


def predict(self, texts,model,device):
    print("\nNormalizing test data...")
    texts = BERTweetNormalizer.normalize_batch(texts)
    dummy_labels = [0] * len(texts)
    test_loader = DataLoader(texts, dummy_labels, shuffle=False)

    model.eval()
    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader, ncols=100):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())

    return [1 if p == 1 else -1 for p in predictions]