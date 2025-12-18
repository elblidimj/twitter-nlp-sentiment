import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, embeddings, hidden_size=128, dropout_rate=0.5):
        super(BiLSTM, self).__init__()
        embedding_dim = embeddings.shape[1]
        print(embedding_dim)
        
        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(embeddings).float(), 
            freeze=False 
        )
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_size, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout_rate
        )
        
        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(embedded)
        
        final_state = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        
        x = torch.relu(self.fc1(self.dropout(final_state)))
        return self.sigmoid(self.fc2(x))
    
def build_lstm(embeddings, hidden_size=128, dropout_rate=0.5):
    return BiLSTM(embeddings, hidden_size, dropout_rate)