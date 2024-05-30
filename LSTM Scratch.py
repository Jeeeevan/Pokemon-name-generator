import pandas as pd
import numpy as np
from torch import nn
import torch.nn.functional as F
import random
import torch
import matplotlib.pyplot as plt
from torch import optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on",device)
features=30
num_epochs=10
hidden_state_size=30
batch_size=16
def load_dataset():
    df = pd.read_csv('pokemon.csv')
    names = df.name.values
    names = [n.lower() for n in names]
    symbols = [' ', '\'', '-', '2', ':', 'é', '♀', '♂']
    cleaned_names = []
    for name in names:
        cleaned_name = ''
        for char in name:
            if char not in symbols:
                cleaned_name += char
        cleaned_names.append(cleaned_name+'.')
    names = cleaned_names
    random.shuffle(names)
    return names
class Embedding(nn.Module):
        def __init__(self,words,features):
            super(Embedding,self).__init__() 
            self.emb_vec = nn.Parameter(torch.randn((words, features)).to(device))
        
        def ret(self,idx):
             return self.emb_vec[idx]
class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,layers=1):
        super(LSTM, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.layers=layers

        self.W_f = nn.ParameterList([nn.Parameter(torch.randn(hidden_size, hidden_size + input_size)).to(device) for _ in range(layers)])
        self.b_f = nn.ParameterList([nn.Parameter(torch.randn(hidden_size)).to(device) for _ in range(layers)])
        
        self.W_i = nn.ParameterList([nn.Parameter(torch.randn(hidden_size, hidden_size + input_size)).to(device) for _ in range(layers)])
        self.b_i = nn.ParameterList([nn.Parameter(torch.randn(hidden_size)).to(device) for _ in range(layers)])
        
        self.W_c = nn.ParameterList([nn.Parameter(torch.randn(hidden_size, hidden_size + input_size)).to(device) for _ in range(layers)])
        self.b_c = nn.ParameterList([nn.Parameter(torch.randn(hidden_size)).to(device) for _ in range(layers)])
        
        self.W_o = nn.ParameterList([nn.Parameter(torch.randn(hidden_size, hidden_size + input_size)).to(device) for _ in range(layers)])
        self.b_o = nn.ParameterList([nn.Parameter(torch.randn(hidden_size)).to(device) for _ in range(layers)])
    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()
        if hidden is None:
            ht = torch.zeros((self.layers, batch_size, self.hidden_size), device=device)
            ct = torch.zeros((self.layers, batch_size, self.hidden_size), device=device)
        else:
            ht, ct = hidden
        output_seq = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            new_ht = torch.zeros((self.layers, batch_size, self.hidden_size), device=device)
            new_ct = torch.zeros((self.layers, batch_size, self.hidden_size), device=device)
            for layer in range(self.layers):
                combined = torch.cat((x_t, ht[layer]), dim=1)
                f_t = torch.sigmoid(self.W_f[layer] @ combined.t() + self.b_f[layer].unsqueeze(1)).t()
                i_t = torch.sigmoid(self.W_i[layer] @ combined.t() + self.b_i[layer].unsqueeze(1)).t()
                o_t = torch.sigmoid(self.W_o[layer] @ combined.t() + self.b_o[layer].unsqueeze(1)).t()
                c_hat_t = torch.tanh(self.W_c[layer] @ combined.t() + self.b_c[layer].unsqueeze(1)).t()
                new_ct[layer] = f_t * ct[layer] + i_t * c_hat_t
                new_ht[layer] = o_t * torch.tanh(new_ct[layer])
                x_t = new_ht[layer]
            output_seq.append(new_ht[-1].unsqueeze(1))
            ht = new_ht
            ct = new_ct
        output_seq = torch.cat(output_seq, dim=1)
        return output_seq, (ht, ct)   
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.fc=nn.Linear(hidden_state_size,27).to(device)
    def forward(self,x):
        out=self.fc(x)
        out=torch.softmax(out,dim=-1)
        return out

def train_model():
    params=list(emb.parameters())+list(lstm.parameters())+list(model.parameters())
    optimizer=torch.optim.Adam(params,lr=0.001)
    criterion=nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()
        for bidx in range(0,len(names),batch_size):
            X=names[bidx:bidx+batch_size]
            optimizer.zero_grad()
            batch_loss = 0
            for name in X:
                hidden = None
                for ch in name:
                    emb_vec=emb.ret(char2idx[ch]).unsqueeze(0).unsqueeze(0)
                    op,hidden=lstm.forward(emb_vec,hidden)
                target = torch.tensor([char2idx[name[-1]]], device=device)
                out=model(op.view(-1))
                loss = criterion(out.unsqueeze(0), target)
                batch_loss += loss
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(names):.4f}')


names=load_dataset()
char2idx = {char: idx for idx, char in enumerate(sorted(set(''.join(names))))}
idx2char = {idx: char for char, idx in char2idx.items()}
emb=Embedding(len(char2idx),features)
lstm=LSTM(features,hidden_state_size,4)
model=Model()
train_model()
# Save the weights of the Embedding, LSTM, and Model
# torch.save(emb.state_dict(), 'embedding_weights.pth')
# torch.save(lstm.state_dict(), 'lstm_weights.pth')
# torch.save(model.state_dict(), 'model_weights.pth')



