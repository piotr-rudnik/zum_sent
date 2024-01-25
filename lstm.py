import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

import torch.nn as nn
import torch.optim as optim


# load models

class SentimentAnalysisLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentAnalysisLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

vocab_size = 17372
embedding_dim = 128
hidden_dim = 256
output_dim = 3

lstm_model = SentimentAnalysisLSTM(vocab_size, embedding_dim, hidden_dim, output_dim)
lstm_model.load_state_dict(torch.load('models/lstm.pth'))


def tokenize(text):
    return text.split()


def collate_batch(batch):
    texts, labels = zip(*batch)
    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return texts, labels

class CustomDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        numericalized_text = [self.vocab.get(word, 0) for word in tokenize(text)]  # 0 dla nieznanych slow
        return torch.tensor(numericalized_text, dtype=torch.long), label

def lstm(zdanie):
    word_counts = Counter()
    word_counts.update(tokenize(zdanie))
    vocab = {word: i+1 for i, word in enumerate(word_counts)} # +1 dla paddingu
    vocab['<pad>'] = 0
    test_dataset = CustomDataset([zdanie], [0], vocab)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_batch)

    lstm_model.eval()
    with torch.no_grad():
        labels_str = ['positive', 'neutral', 'negative']
        for texts, labels in test_loader:
            outputs = lstm_model(texts)
            _, predicted = torch.max(outputs.data, 1)
            return labels_str[predicted.item()]
            # print("OUTPUT: ", predicted.item())