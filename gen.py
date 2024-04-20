import torch
import numpy as np
from train import NN,imap
context_size = 4
epochs = 1000
feat = 12
batch_size = 32

def generate_names(model_path, EM):
    model = NN(EM)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    generated_names = []
    for _ in range(10):
        rr = np.random.randint(1, 27)
        context = [0] * (context_size - 1) + [rr]
        pkm = [imap[rr]]
        check = 0
        while context[-1] != 0:
            check += 1
            if check > 10:
                break
            emb = EM[[context]]
            r = model(emb.view(-1, context_size * feat))
            r = torch.argmax(r).item()
            context = context[1:] + [r]
            pkm.append(imap[r])
        generated_names.append(''.join(pkm))
    return generated_names

if __name__ == "__main__":
    model_path = 'weights.pt'  # Path to the saved model weights
    EM = torch.randn((27, feat))  # Assuming the same embedding matrix as used during training
    generated_names = generate_names(model_path, EM)
    for name in generated_names:
        print(name)
