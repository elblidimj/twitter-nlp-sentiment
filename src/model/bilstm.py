import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, embeddings, hidden_size=128, dropout_rate=0.5):
        super(BiLSTM, self).__init__()
        # Dynamically determine size to prevent out-of-bounds CUDA errors
        vocab_size, embedding_dim = embeddings.shape
        
        # Initialize embedding layer with pre-trained weights
        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(embeddings).float(), 
            freeze=False 
        )
        
        # Two-layer Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_size, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout_rate if 2 > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        # hidden_size * 2 because the LSTM is bidirectional
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(embedded)
        
        # Concatenate final hidden states from forward and backward passes
        # hn shape: (num_layers * 2, batch, hidden_size)
        final_state = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        
        x = torch.relu(self.fc1(self.dropout(final_state)))
        return self.sigmoid(self.fc2(x))