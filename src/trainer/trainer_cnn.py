import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from src.model.cnn import TwitterCNN

def train_cnn(X, y, model, device, embeddings, lr, epochs=6):
    vocab_size = embeddings.shape[0]
    X = np.clip(X, 0, vocab_size - 1)    
    y_pt = np.where(y == 1, 1, 0)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    full_loader = DataLoader(
        TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(y_pt).float()), 
        batch_size=512, shuffle=True
    )

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for bx, by in full_loader:
            bx, by = bx.to(device), by.to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(bx)
            loss = nn.BCELoss()(outputs, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"CNN Epoch {epoch+1}/{epochs} | Avg Loss: {epoch_loss/len(full_loader):.4f}")
    return model

def predict_cnn(model, device, X_test, embeddings):
    vocab_size = embeddings.shape[0]
    X_test = np.clip(X_test, 0, vocab_size - 1)
    
    model.eval()
    with torch.no_grad():
        test_inputs = torch.from_numpy(X_test).long().to(device)
        test_probs = model(test_inputs).cpu().numpy()
        predictions = np.where(test_probs > 0.5, 1, -1).astype(int).flatten()
    return predictions

def grid_cnn(X_train, y_train, X_val, y_val, embeddings, device):
    y_train_pt = np.where(y_train == 1, 1, 0)
    y_val_pt = np.where(y_val == 1, 1, 0)

    param_grid = {
        "learning_rate": [0.001, 0.0005],
        "kernel_size": [3, 5],
        "filters": [64, 128],
    }

    best_f1 = 0.0
    best_config = None

    for lr in param_grid["learning_rate"]:
        for k in param_grid["kernel_size"]:
            for f in param_grid["filters"]:
                print(f"\nEvaluating CNN: LR={lr}, K={k}, F={f}")
                model = TwitterCNN(embeddings, kernel_size=k, filters=f).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                
                train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).long(), torch.from_numpy(y_train_pt).float()), batch_size=512, shuffle=True)
                val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val).long(), torch.from_numpy(y_val_pt).float()), batch_size=512)

                for _ in range(2):
                    model.train()
                    for bx, by in train_loader:
                        bx, by = bx.to(device), by.to(device).view(-1, 1)
                        optimizer.zero_grad()
                        nn.BCELoss()(model(bx), by).backward()
                        optimizer.step()

                model.eval()
                val_preds = []
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx = bx.to(device)
                        outputs = model(bx).cpu().numpy()
                        val_preds.extend((outputs > 0.5).astype(int).flatten())

                current_f1 = f1_score(y_val_pt, val_preds)
                print(f"F1 Score: {current_f1:.4f}")
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_config = (lr, k, f)

    print(f"Best Configuration: LR={best_config[0]}, K={best_config[1]}, F={best_config[2]}")
    return best_config