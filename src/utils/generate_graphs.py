import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

CSV_PATH = "cnn_experiment_results.csv"
OUTPUT_DIR = "cnn_report_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_report_visuals(csv_path):
    # 1. Load Data
    df = pd.read_csv(csv_path)

    # 2. Extract Hyperparameters from Config_Name
    # Pattern: LR_0.001_K_3_F_64_D_0.3
    regex = r'LR_(?P<LR>[\d\.]+)_K_(?P<Kernel_Size>\d+)_F_(?P<Filters>\d+)_D_(?P<Dropout>[\d\.]+)'
    extracted = df['Config_Name'].str.extract(regex)
    
    df['LR'] = pd.to_numeric(extracted['LR'])
    df['Kernel_Size'] = pd.to_numeric(extracted['Kernel_Size'])
    df['Filters'] = pd.to_numeric(extracted['Filters'])
    df['Dropout'] = pd.to_numeric(extracted['Dropout'])

    # 3. Calculate Performance Metrics
    df['Precision'] = df['TP'] / (df['TP'] + df['FP'] + 1e-9)
    df['Recall'] = df['TP'] / (df['TP'] + df['FN'] + 1e-9)
    df['F1_Score'] = 2 * (df['Precision'] * df['Recall']) / (df['Precision'] + df['Recall'] + 1e-9)

    sns.set_theme(style="whitegrid", palette="muted")

    # --- PLOT 1: Hyperparameter Heatmap ---
    final_epoch = df[df['Epoch'] == df['Epoch'].max()]
    pivot = final_epoch.pivot_table(index='Filters', columns='Kernel_Size', values='Accuracy', aggfunc='mean')

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.4f', cbar_kws={'label': 'Mean Accuracy'})
    plt.title('CNN Architecture Impact: Filters vs. Kernel Size', fontsize=14, pad=15)
    plt.savefig(f"{OUTPUT_DIR}/heatmap_architecture.png", bbox_inches='tight')
    plt.close()

    # --- PLOT 2: Stability Boxplot (F1-Score) ---
    plt.figure(figsize=(12, 6))
    final_epoch['Short_Name'] = final_epoch['Config_Name'].apply(lambda x: x.replace('LR_', 'lr').replace('_K_', ' k').replace('_F_', ' f').replace('_D_', ' d'))
    sns.boxplot(x='Short_Name', y='F1_Score', data=final_epoch)
    plt.xticks(rotation=45, ha='right')
    plt.title('F1-Score Consistency across K-Folds', fontsize=14)
    plt.ylabel('F1-Score (Validation)')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/boxplot_stability.png", bbox_inches='tight')
    plt.close()

    # --- PLOT 3: Learning Curve (Best Config) ---
    best_config_name = final_epoch.groupby('Config_Name')['Accuracy'].mean().idxmax()
    best_df = df[df['Config_Name'] == best_config_name]
    curves = best_df.groupby('Epoch').mean(numeric_only=True).reset_index()

    plt.figure(figsize=(10, 5))
    plt.plot(curves['Epoch'], curves['Train_Loss'], marker='o', color='#e74c3c', linewidth=2, label='Training Loss')
    plt.title(f'Learning Curve: Top-Performing CNN\n({best_config_name})', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/learning_curve_best.png", bbox_inches='tight')
    plt.close()

    # --- PLOT 4: Aggregated Confusion Matrix ---
    best_totals = final_epoch[final_epoch['Config_Name'] == best_config_name].sum(numeric_only=True)
    cm = np.array([[best_totals['TN'], best_totals['FP']],
                   [best_totals['FN'], best_totals['TP']]])

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Aggregated Confusion Matrix (Best Config)', fontsize=14, pad=15)
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('Actual Sentiment')
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_best.png", bbox_inches='tight')
    plt.close()

    print(f" Report plots generated in: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_report_visuals(CSV_PATH)