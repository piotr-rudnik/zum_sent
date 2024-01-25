import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

file_path = 'data.csv'
data = pd.read_csv(file_path).head(50000)
data = data.drop(columns=['index'])
data['tweets'] = data['tweets'].str.replace('[^a-zA-Z\s]', '').str.lower()

def load_glove_embeddings(path):
    embeddings_dict = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float)
            embeddings_dict[word] = vector
    return embeddings_dict

glove_path = 'glove.6B.50d.txt'  # Update this path
glove_embeddings = load_glove_embeddings(glove_path)

#%%

# Prepare Data
X_train, X_test, y_train, y_test = train_test_split(data['tweets'], data['labels'], test_size=0.2, random_state=42)

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Build Vocabulary
vocab = {"<PAD>": 0}
for text in X_train:
    for word in text.split():
        if word not in vocab:
            vocab[word] = len(vocab)

embedding_dim = 50
embedding_matrix = torch.zeros((len(vocab), embedding_dim))
for word, idx in vocab.items():
    embedding_vector = glove_embeddings.get(word)
    if embedding_vector is not None:
        embedding_matrix[idx] = embedding_vector
    else:
        embedding_matrix[idx] = torch.randn(embedding_dim)  # Random vector for unknown words

# Model
class SimpleNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)  # Average pooling
        return self.fc(pooled)

# Instantiate model, loss, optimizer
model = SimpleNN(len(vocab), embedding_dim, len(le.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# load model weghts
model.load_state_dict(torch.load('models/glove.pth'))

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    texts, labels = zip(*batch)
    # Pad the sequences to the maximum length in the batch
    texts = pad_sequence(texts, batch_first=True, padding_value=vocab["<PAD>"])
    labels = torch.tensor(labels, dtype=torch.long)
    return texts, labels



def glove(zdanie):
    model.eval()
    with torch.no_grad():
        tokens = [vocab.get(word, 0) for word in zdanie.split()]
        tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        prediction = model(tokens_tensor)
        _, predicted_label = torch.max(prediction, dim=1)
        return le.classes_[predicted_label]