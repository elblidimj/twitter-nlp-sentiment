from sklearn.model_selection import train_test_split
from src.model.bertweet import build_model
from src.trainer.bertweet_train import train, predict
from src.datasets.twitter import load_training_tweets, load_test_tweets
import pandas as pd
import torch

print('Loading data...')
train_texts, train_labels = load_training_tweets(use_full=False)
test_texts, test_ids = load_test_tweets()
X_train, X_val, y_train, y_val = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = build_model(device)
print('Training...')
train(X_train, y_train, X_val, y_val,device = device)


predictions = predict(test_texts,model=model,device=device)

pd.DataFrame({'Id': test_ids, 'Prediction': predictions}).to_csv('submission.csv', index=False)
