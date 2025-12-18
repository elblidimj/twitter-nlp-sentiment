import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import os
import itertools
from src.model.cnn import TwitterCNN

SAVE_PATH = "twitter-datasets"

def train_cnn(X, y, model, device, embeddings, lr, epochs=3):
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


### Code for Cross-Validation obtaining the bast parameters for CNN
def grid_cnn(X, y, embeddings, device):
    """
    Performs 3-Fold Cross-Validation over a hyperparameter grid.
    Returns: best_lr, best_k, best_f, best_d
    """
    vocab_size = embeddings.shape[0]
    X = np.clip(X, 0, vocab_size - 1)
    y_pt = np.where(y == 1, 1, 0)

    param_grid = {
        'learning_rate': [0.001, 0.0005],
        'kernel_size': [3, 5],
        'filters': [64, 128],
        'dropout': [0.3, 0.5]
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    performance_log = []
    best_acc = 0.0
    best_config = None

    print("--- Starting CNN 3-Fold Cross-Validation ---")

    for config in combinations:
        config_name = f"LR_{config['learning_rate']}_K_{config['kernel_size']}_F_{config['filters']}_D_{config['dropout']}"
        print(f"\nEvaluating: {config_name}")
        
        kfold = KFold(n_splits=3, shuffle=True, random_state=42)
        dataset = TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(y_pt).float())

        for fold, (train_ids, val_ids) in enumerate(kfold.split(X)):
            train_loader = DataLoader(Subset(dataset, train_ids), batch_size=1024, shuffle=True)
            val_loader = DataLoader(Subset(dataset, val_ids), batch_size=1024)

            model = TwitterCNN(
                embeddings, 
                kernel_size=config['kernel_size'], 
                filters=config['filters'], 
                dropout_rate=config['dropout']
            ).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
            criterion = nn.BCELoss()

            # Short training for grid search (2 epochs)
            for epoch in range(2):
                model.train()
                for bx, by in train_loader:
                    bx, by = bx.to(device), by.to(device).view(-1, 1)
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    optimizer.step()
                
                # Validation Metrics after each epoch
                model.eval()
                tp, tn, fp, fn = 0, 0, 0, 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(device), vy.to(device).view(-1, 1)
                        v_out = model(vx)
                        v_pred = (v_out > 0.5).float()
                        tp += ((v_pred == 1) & (vy == 1)).sum().item()
                        tn += ((v_pred == 0) & (vy == 0)).sum().item()
                        fp += ((v_pred == 1) & (vy == 0)).sum().item()
                        fn += ((v_pred == 0) & (vy == 1)).sum().item()
                
                acc = (tp + tn) / (tp + tn + fp + fn)
                performance_log.append({
                    'Config_Name': config_name, 'Fold': fold + 1, 'Epoch': epoch + 1,
                    'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'Accuracy': acc
                })

        # Calculate mean accuracy for this config across all folds and epochs
        current_mean = np.mean([entry['Accuracy'] for entry in performance_log if entry['Config_Name'] == config_name])
        print(f"Mean CV Accuracy: {current_mean:.4f}")
        
        if current_mean > best_acc:
            best_acc = current_mean
            best_config = (config['learning_rate'], config['kernel_size'], config['filters'], config['dropout'])

    # Save log for visualization
    os.makedirs(SAVE_PATH, exist_ok=True)
    pd.DataFrame(performance_log).to_csv(os.path.join(SAVE_PATH, "cnn_experiment_results.csv"), index=False)
    
    print(f"\ Best CV Config: LR={best_config[0]}, K={best_config[1]}, F={best_config[2]}, D={best_config[3]}")
    return best_config