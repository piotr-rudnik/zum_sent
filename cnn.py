from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

device = torch.device('cpu')

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = self.embedding(x)  # [batch size, sent len, emb dim]
        x = x.unsqueeze(1)  # [batch size, 1, sent len, emb dim]
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in x]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x)


from sklearn.model_selection import train_test_split
###############
# ładowanie danych z csv i tworzenie datasetów
# dla pierwszych 3 modeli bierzemy tylko 50k rekordów, inaczej proces uczenia jest za długi
###############
file_path = 'data.csv'
data = pd.read_csv(file_path).head(50000)
data = data.drop(columns=['index'])
data['tweets'] = data['tweets'].str.replace('[^a-zA-Z\s]', '').str.lower()

X_train, X_test, y_train, y_test = train_test_split(data['tweets'], data['labels'], test_size=0.2, random_state=42)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)




#%%

###############
# Enkodowanie i tokenizacja
###############
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)


def tokenize(text):
    return text.split()

word_counts = Counter()
for text in X_train:
    word_counts.update(tokenize(text))
vocab = {word: i+1 for i, word in enumerate(word_counts)} # +1 dla paddingu
vocab['<pad>'] = 0

vocab_size = len(vocab)
embedding_dim = 100
n_filters = 100
filter_sizes = [2, 3, 4]
output_dim = 3
dropout = 0.3

model = TextCNN(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout).to(device)


optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()




def cnn(zdanie):
    model.eval()
    with torch.no_grad():
        tokens = [vocab.get(word, 0) for word in zdanie.split()]
        tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        prediction = model(tokens_tensor)
        _, predicted_label = torch.max(prediction, dim=1)
        return le.inverse_transform(predicted_label)[0]
