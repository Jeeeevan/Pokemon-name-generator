import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random 
import numpy as np
random.seed(4)

class NN(nn.Module):
    def __init__(self) -> None:
        super(NN,self).__init__()
        self.fc1=nn.Linear(3,16)
        self.fc2=nn.Linear(16,128)
        self.fc4=nn.Linear(128,27)
        self.do=nn.Dropout(0.5)
    def forward(self, x):
        x = x.float()  
        x = F.tanh(self.fc1(x))
        x=self.do(x)
        x = F.tanh(self.fc2(x))
        x=self.do(x)
        x = self.fc4(x)
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
        short_term_memory=[0]*3
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

model=NN()
lossfn=nn.CrossEntropyLoss()   
opt=optim.SGD(model.parameters(),lr=0.001)
epochs=1
# print(Ytrain[0].float())
for epoch in range(epochs):
    bno=torch.randint(0,Xtrain.shape[0],(32,))
    model.train()
    running_loss=0.0    
    running_acc=0.0
    input=Xtrain[bno]
    label=Ytrain[bno]
    for i in range(len(input)):
        out=model(input[i].unsqueeze(0))
        loss=lossfn(out, label[i].unsqueeze(0))
        opt.zero_grad()
        loss.backward()
        opt.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(Xtrain)}")
model.eval()
# test_loss = 0.0
# with torch.no_grad():
#     for i in range(len(Xtest)):
#         out = model(Xtest[i].unsqueeze(0))
#         test_loss += lossfn(out, Ytest[i].unsqueeze(0)).item()

# print(f"Test Loss: {test_loss / len(Xtest)}")
for j in range(5):
    rr=random.randint(1,27)
    context=[0,0,rr]
    pkm=[imap[rr]]
    check=0
    while context[-1]!=0:
        check=check+1
        if check>10:
            break
        r=model(torch.Tensor(context))
        r=torch.argmax(r).item()
        context=context[1:]+[r]
        pkm.append(imap[r])
    print(''.join(pkm))

