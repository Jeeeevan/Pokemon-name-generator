import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random 
import numpy as np
random.seed(4)
context_size=4
epochs=1000
feat=6
batch_size=32

class NN(nn.Module):
    def __init__(self,EM) -> None:
        super(NN,self).__init__()
        self.fc1=nn.Linear(context_size*feat,16)
        self.bn1=nn.BatchNorm1d(1)
        self.fc2=nn.Linear(16,27)
        self.EM=nn.Parameter(EM)
    def forward(self, x):
        x=x.float()
        x=F.tanh(self.fc1(x))
        # x=self.bn1(x)
        x=F.tanh(self.fc2(x))
        return x
def load_dataset():
    df=pd.read_csv('pokemon.csv')
    names=df.name.values
    names=[n.lower() for n in names ]
    symbols = [' ', '\'', '-', '2', ':', 'é', '♀', '♂']
    cleaned_names = []
    for name in names:
        cleaned_name = ''
        for char in name:
            if char not in symbols:
                cleaned_name += char
        cleaned_names.append(cleaned_name)
    names = cleaned_names
    split=int(0.9*len(names))
    random.shuffle(names)
    train=names[:split]
    test=names[split:]
    return names,train,test

def create_dataset(words):
    X, Y = [], []
    for w in words:
        short_term_memory=[0]*context_size
        for ch in w+'.':    
            X.append(short_term_memory)
            Y.append(cmap[ch])
            short_term_memory=short_term_memory[1:]+[cmap[ch]]
    return torch.tensor(X),torch.tensor(Y)

names,train,test=load_dataset()
char=[ele for ele in ''.join(names) if ele not in [' ','\'','-','2',':','é', '♀', '♂']]
char=sorted(list(set(char)))
cmap={ele:i for i,ele in enumerate(char)}
imap={i:ele for i,ele in enumerate(char)}
Xtrain,Ytrain=create_dataset(train)
Xtest,Ytest=create_dataset(test)

EM=torch.randn((27,feat))
model=NN(EM)
lossfn=nn.CrossEntropyLoss()   
opt=optim.SGD(model.parameters(),lr=0.001)

print("Training started")
for epoch in range(epochs):
    model.train()
    running_loss=0.0    
    for batchidx in range(0,len(Xtrain),batch_size):
        input=Xtrain[batchidx:batchidx+batch_size]
        label=Ytrain[batchidx:batchidx+batch_size]

        emb=EM[input]
        out=model(emb.view(-1,context_size*feat))
        loss=lossfn(out, label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        running_loss += loss.item()
    if epoch%20==0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(Xtrain)}")
model.eval()
print("End of training")
torch.save(model.state_dict(),'weights.pt')
print("Weights saved")
test_loss = 0.0
with torch.no_grad():
    for i in range(len(Xtest)):
        out = model(EM[Xtest].view(-1,context_size*feat))
        test_loss += lossfn(out, Ytest).item()

print(f"Test Loss: {test_loss / len(Xtest)}")
for j in range(5):
    rr=random.randint(1,27)
    context=[0]*(context_size-1)+[rr]
    pkm=[imap[rr]]
    check=0
    while context[-1]!=0:
        check=check+1
        if check>10:
            break
        emb=EM[[context]]
        r=model(emb.view(-1,context_size*feat))
        r=torch.argmax(r).item()
        context=context[1:]+[r]
        pkm.append(imap[r])
    print(''.join(pkm))

