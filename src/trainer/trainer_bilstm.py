import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import os
import itertools
from src.model.bilstm import BiLSTM

SAVE_PATH = "twitter-datasets"

def train_lstm(X, y, device, embeddings,hidden_units,dropout_rate,lr,epochs=3):
    
    vocab_size = embeddings.shape[0]
    X = np.clip(X, 0, vocab_size - 1)    
    y_pt = np.where(y == 1, 1, 0)

    final_model = BiLSTM(
        embeddings, 
        hidden_size=hidden_units, 
        dropout_rate=dropout_rate
    ).to(device)
    
    optimizer = optim.Adam(final_model.parameters(), lr=lr)
    full_loader = DataLoader(
        TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(y_pt).float()), 
        batch_size=512, shuffle=True
    )

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
        print(f"Final Epoch {epoch+1}/5 | Avg Loss: {epoch_loss/len(full_loader):.4f}")

    return final_model

def grid_lstm(X,y_pt,embeddings, device):
    SAVE_PATH = "../../data"
    param_grid = {
        'learning_rate': [0.001, 0.0005],
        'hidden_units': [64, 128],
        'dropout': [0.3, 0.5]
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_acc = 0
    best_config = None
    performance_log = []

    print(f"--- Starting Grid Search & Detailed Logging over {len(combinations)} combinations ---")

    for config in combinations:
        config_name = f"LR_{config['learning_rate']}_H_{config['hidden_units']}_D_{config['dropout']}"
        print(f"\nEvaluating Config: {config_name}")
        
        kfold = KFold(n_splits=3, shuffle=True, random_state=42)

        for fold, (train_ids, val_ids) in enumerate(kfold.split(X)):
            # Prepare DataLoaders
            train_loader = DataLoader(
                Subset(TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(y_pt).float()), train_ids), 
                batch_size=1024, shuffle=True
            )
            val_loader = DataLoader(
                Subset(TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(y_pt).float()), val_ids), 
                batch_size=1024
            )

            model = BiLSTM(
                embeddings, 
                hidden_size=config['hidden_units'], 
                dropout_rate=config['dropout']
            ).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
            criterion = nn.BCELoss()

            for epoch in range(2):
                model.train()
                total_train_loss = 0
                for bx, by in train_loader:
                    bx, by = bx.to(device), by.to(device).view(-1, 1)
                    optimizer.zero_grad()
                    outputs = model(bx)
                    loss = criterion(outputs, by)
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()

                model.eval()
                tp, tn, fp, fn = 0, 0, 0, 0
                val_loss = 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(device), vy.to(device).view(-1, 1)
                        v_out = model(vx)
                        val_loss += criterion(v_out, vy).item()
                        
                        v_pred = (v_out > 0.5).float()
                        tp += ((v_pred == 1) & (vy == 1)).sum().item()
                        tn += ((v_pred == 0) & (vy == 0)).sum().item()
                        fp += ((v_pred == 1) & (vy == 0)).sum().item()
                        fn += ((v_pred == 0) & (vy == 1)).sum().item()
                
                performance_log.append({
                    'Config_Name': config_name,
                    'Fold': fold + 1,
                    'Epoch': epoch + 1,
                    'Learning_Rate': config['learning_rate'],
                    'Hidden_Units': config['hidden_units'],
                    'Dropout': config['dropout'],
                    'Train_Loss': total_train_loss / len(train_loader),
                    'Val_Loss': val_loss / len(val_loader),
                    'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
                    'Accuracy': (tp + tn) / (tp + tn + fp + fn)
                })

        current_mean = np.mean([entry['Accuracy'] for entry in performance_log if entry['Config_Name'] == config_name])
        print(f"Config Mean Val Acc: {current_mean:.4f}")
        if current_mean > best_acc:
            best_acc = current_mean
            best_config = config

    log_df = pd.DataFrame(performance_log)
    if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)
    log_df.to_csv(os.path.join(SAVE_PATH, "bilstm_experiment_results.csv"), index=False)
    print(f"\n🏆 Best Config: {best_config} | Master log saved to {SAVE_PATH}/bilstm_experiment_results.csv")

