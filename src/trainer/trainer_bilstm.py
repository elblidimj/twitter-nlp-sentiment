import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.model.bilstm import BiLSTM
from helpers import create_csv_submission

SAVE_PATH = "twitter-datasets"

def train_lstm(X, y, model,device, embeddings,lr,epochs=3):
    
    vocab_size = embeddings.shape[0]
    X = np.clip(X, 0, vocab_size - 1)    
    y_pt = np.where(y == 1, 1, 0)

    final_model = model
    
    optimizer = optim.Adam(final_model.parameters(), lr=lr)
    full_loader = DataLoader(
        TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(y_pt).float()), 
        batch_size=64, shuffle=True
    )
    best_f1 = 0
    patience = 2
    bad_epochs = 0

    for epoch in range(epochs):
        final_model.train()
        epoch_loss = 0
        for bx, by in full_loader:
            bx, by = bx.to(device), by.to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = final_model(bx)
            loss = nn.BCELoss()(outputs, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Final Epoch {epoch+1}/{epochs} | Avg Loss: {epoch_loss/len(full_loader):.4f}")

    return final_model

def predict_lstm(model, device, test_ids,X_test,embeddings):
    vocab_size = embeddings.shape[0]
    

    X_test = np.clip(X_test, 0, vocab_size - 1)
    
    model.eval()
    with torch.no_grad():
        test_inputs = torch.from_numpy(X_test).long().to(device)
        test_probs = model(test_inputs).cpu().numpy()
        y_test_pred = np.where(test_probs > 0.5, 1, -1).astype(int)

    return y_test_pred

def grid_lstm(X_train, y_train, X_val, y_val, embeddings, device):
    y_train_pt = np.where(y_train == 1, 1, 0)
    y_val_pt   = np.where(y_val == 1, 1, 0)

    param_grid = {
        "learning_rate": [0.001, 0.0005],
        "hidden_units": [64, 128],
        "dropout": [0.3, 0.5],
    }

    best_f1 = 0.0
    best_config = None
    logs = []

    for lr in param_grid["learning_rate"]:
        for hidden in param_grid["hidden_units"]:
            for dropout in param_grid["dropout"]:

                print(f"\nEvaluating LR={lr}, H={hidden}, D={dropout}")

                model = BiLSTM(
                    embeddings,
                    hidden_size=hidden,
                    dropout_rate=dropout
                ).to(device)

                optimizer = optim.Adam(model.parameters(), lr=lr)
                criterion = nn.BCELoss()

                train_loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(X_train).long(),
                        torch.from_numpy(y_train_pt).float()
                    ),
                    batch_size=512,
                    shuffle=True
                )

                val_loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(X_val).long(),
                        torch.from_numpy(y_val_pt).float()
                    ),
                    batch_size=512
                )

                for epoch in range(3):
                    model.train()
                    for bx, by in train_loader:
                        bx = bx.to(device)
                        by = by.to(device).view(-1, 1)

                        optimizer.zero_grad()
                        loss = criterion(model(bx), by)
                        loss.backward()
                        optimizer.step()

                model.eval()
                val_preds = []
                val_targets = []

                with torch.no_grad():
                    for bx, by in val_loader:
                        bx = bx.to(device)
                        outputs = model(bx).cpu().numpy()
                        preds = (outputs > 0.5).astype(int).flatten()

                        val_preds.extend(preds)
                        val_targets.extend(by.numpy())

                val_acc = accuracy_score(val_targets, val_preds)
                val_f1  = f1_score(val_targets, val_preds)

                logs.append({
                    "lr": lr,
                    "hidden": hidden,
                    "dropout": dropout,
                    "val_acc": val_acc,
                    "val_f1": val_f1
                })

                print(f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_config = (lr, hidden, dropout)

    print(f"\n🏆 Best config: LR={best_config[0]}, "
          f"H={best_config[1]}, D={best_config[2]} | F1={best_f1:.4f}")

    return best_config

