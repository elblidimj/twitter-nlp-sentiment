import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import os
import itertools

# Custom internal imports
from helpers import create_csv_submission
from src.utils.io_utils import load_vocab_and_embeddings
from src.datasets.twitter import load_training_tweets, load_test_tweets
from src.transforms.text_embeddings import tweets_to_matrix
from src.model.bilstm import BiLSTM

SAVE_PATH = "twitter-datasets"

def train_and_predict_bilstm(data_dir="twitter-datasets"):
    # --- GPU SETUP ---
    # Checks if a compatible NVIDIA GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    vocab, embeddings = load_vocab_and_embeddings("vocab.pkl", "embeddings.npy")
    tweets, y = load_training_tweets(data_dir=data_dir, use_full=True)
    X = tweets_to_matrix(tweets, vocab, embeddings, None)
    
    # --- CUDA FIX: Index Clipping ---
    # Ensures all word indices are within the embedding matrix range to prevent GPU crashes
    vocab_size = embeddings.shape[0]
    X = np.clip(X, 0, vocab_size - 1)
    print(f"Indices clipped to range [0, {vocab_size - 1}]")
    
    # Convert labels from -1/1 to 0/1 for Binary Cross Entropy
    y_pt = np.where(y == 1, 1, 0)

    # 2. Hyperparameter Grid Search Setup
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

            # Initialize model with specific hyperparameters
            model = BiLSTM(
                embeddings, 
                hidden_size=config['hidden_units'], 
                dropout_rate=config['dropout']
            ).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
            criterion = nn.BCELoss()

            # Train for comparison epochs
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

                # --- Validation & Raw Metric Extraction ---
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
                
                # Log detailed stats for CSV
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

        # Calculate mean for best config tracking
        current_mean = np.mean([entry['Accuracy'] for entry in performance_log if entry['Config_Name'] == config_name])
        print(f"Config Mean Val Acc: {current_mean:.4f}")
        if current_mean > best_acc:
            best_acc = current_mean
            best_config = config

    # 3. Save the Master CSV Log
    log_df = pd.DataFrame(performance_log)
    if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)
    log_df.to_csv(os.path.join(SAVE_PATH, "bilstm_experiment_results.csv"), index=False)
    print(f"\n🏆 Best Config: {best_config} | Master log saved to {SAVE_PATH}/bilstm_experiment_results.csv")

    # 4. Final Training on FULL Dataset with Best Params
    print("\n--- Final Training on Full Dataset ---")
    final_model = BiLSTM(
        embeddings, 
        hidden_size=best_config['hidden_units'], 
        dropout_rate=best_config['dropout']
    ).to(device)
    
    optimizer = optim.Adam(final_model.parameters(), lr=best_config['learning_rate'])
    full_loader = DataLoader(
        TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(y_pt).float()), 
        batch_size=512, shuffle=True
    )

    for epoch in range(5):
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

    # 5. Generate Submission
    print("\nGenerating final submission file...")
    test_ids, test_tweets = load_test_tweets(data_dir=data_dir)
    X_test = tweets_to_matrix(test_tweets, vocab, embeddings, None)
    
    # Apply index clipping to test data as well
    X_test = np.clip(X_test, 0, vocab_size - 1)
    
    final_model.eval()
    with torch.no_grad():
        test_inputs = torch.from_numpy(X_test).long().to(device)
        test_probs = final_model(test_inputs).cpu().numpy()
        y_test_pred = np.where(test_probs > 0.5, 1, -1).astype(int)

    create_csv_submission(test_ids, y_test_pred, "submission_bilstm_final.csv")
    print("✅ Process Complete. Submission saved as 'submission_bilstm_final.csv'")

if __name__ == "__main__":
    train_and_predict_bilstm()