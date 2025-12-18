import torch
import torch.nn as nn

class TwitterCNN(nn.Module):
    def __init__(self, embeddings, kernel_size=3, filters=128, dropout_rate=0.5):
        super(TwitterCNN, self).__init__()
        vocab_size, embedding_dim = embeddings.shape
        
        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(embeddings).float(), 
            freeze=False 
        )
        
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=filters, kernel_size=kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(filters)
        self.conv2 = nn.Conv1d(in_channels=filters, out_channels=128, kernel_size=5, padding='same')
        
        self.pool = nn.AdaptiveMaxPool1d(1) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        
        x = self.relu(self.fc1(self.dropout(x)))
        return self.sigmoid(self.fc2(x))

def build_cnn(embeddings, kernel_size=3, filters=128, dropout_rate=0.5):
    return TwitterCNN(embeddings, kernel_size, filters, dropout_rate)