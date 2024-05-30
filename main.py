import pandas as pd
import numpy as np
from torch import nn
import torch.nn.functional as F
import random
import torch
import matplotlib.pyplot as plt
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)

features = 30
num_epochs = 100
hidden_state_size = 30
batch_size = 16

def load_dataset():
    df = pd.read_csv('pokemon.csv')
    names = df.name.values
    names = [n.lower() for n in names]
    symbols = [' ', '\'', '-', '2', ':', 'é', '♀', '♂']
    cleaned_names = []
    for name in names:
        cleaned_name = ''.join([char for char in name if char not in symbols])
        cleaned_names.append(cleaned_name+'.')
    random.shuffle(cleaned_names)
    return cleaned_names

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(device)
        
    def forward(self, idx):
        return self.embedding(idx)

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(Model, self).__init__()
        self.embedding = Embedding(output_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)
    
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)  # Apply the linear layer to each time step
        return out, hidden
    
    def generate(self, char2idx, idx2char, start_char='a', max_length=20):
        idx = torch.tensor([char2idx[start_char]], dtype=torch.long).unsqueeze(0).to(device)
        name = start_char
        hidden = None
        for _ in range(max_length - 1):
            output, hidden = self.forward(idx, hidden)
            output = F.softmax(output[:, -1, :], dim=-1)  # Softmax on the last time step's output
            top_i = torch.multinomial(output, 1).item()
            next_char = idx2char[top_i]
            if next_char == '.':  # Assuming '<end>' is a special end token if used
                break
            name += next_char
            idx = torch.tensor([top_i], dtype=torch.long).unsqueeze(0).to(device)
        return name

def train_model(model, names, char2idx, idx2char, num_epochs, batch_size, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        epoch_loss = 0
        random.shuffle(names)
        
        for bidx in range(0, len(names), batch_size):
            batch_names = names[bidx:bidx + batch_size]
            max_length = max(len(name) for name in batch_names)
            input_batch = torch.zeros(batch_size, max_length - 1, dtype=torch.long).to(device)
            target_batch = torch.zeros(batch_size, max_length - 1, dtype=torch.long).to(device)
            
            for i, name in enumerate(batch_names):
                for t in range(len(name) - 1):
                    input_batch[i, t] = char2idx[name[t]]
                    target_batch[i, t] = char2idx[name[t + 1]]
            
            optimizer.zero_grad()
            output, _ = model(input_batch)
            output = output.view(-1, vocab_size)  # Reshape for loss computation
            target_batch = target_batch.view(-1)
            loss = criterion(output, target_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(names):.4f}')

names = load_dataset()
char2idx = {char: idx for idx, char in enumerate(sorted(set(''.join(names))))}
idx2char = {idx: char for char, idx in char2idx.items()}

vocab_size = len(char2idx)
model = Model(features, hidden_state_size, vocab_size, num_layers=2)

train_model(model, names, char2idx, idx2char, num_epochs, batch_size)

# Save the weights of the Model
torch.save(model.state_dict(), 'model_weights.pth')

# Load the model weights (optional, for demonstration)
# model.load_state_dict(torch.load('model_weights.pth'))

# Generate new names
for _ in range(10):
    start_char = random.choice(list(char2idx.keys()))
    print(model.generate(char2idx, idx2char, start_char))
