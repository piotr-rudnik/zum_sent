import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from collections import Counter

###############
# Load data
###############
file_path = 'file.csv'
data = pd.read_csv(file_path).head(5000)
data = data.drop(columns=['index'])
data['tweets'] = data['tweets'].str.replace('[^a-zA-Z\s]', '').str.lower()

###############
# Prepare datasets
###############
X_train, X_test, y_train, y_test = train_test_split(data['tweets'], data['labels'], test_size=0.2, random_state=42)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


###############
# Encoding lavels
###############
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

###############
# Making tokens
###############
def tokenize(text):
    return text.split()

word_counts = Counter()
for text in X_train:
    word_counts.update(tokenize(text))
vocab = {word: i+1 for i, word in enumerate(word_counts)} # +1 dla paddingu
vocab['<pad>'] = 0

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

def collate_batch(batch):
    text_list, labels = zip(*batch)
    text_tensor = pad_sequence([text for text, _ in batch], batch_first=True, padding_value=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return text_tensor, labels_tensor

# Create DataLoader instances
batch_size = 256
train_dataset = CustomDataset(X_train, y_train_encoded, vocab)
test_dataset = CustomDataset(X_test, y_test_encoded, vocab)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
