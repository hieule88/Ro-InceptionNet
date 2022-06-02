import numpy as np
import matplotlib.pyplot as plt
from  tqdm import tqdm
import torch
import torchvision
from torchvision.utils import make_grid
from model import Model, InceptionV4, rand_bbox
from loaddata import Dataset

def main():
    dataset = Dataset()
    train_ds = dataset.train_ds
    train_dl = dataset.dataloader(20, train=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr =  0.0001
    epochs = 30

    model = InceptionV4()
    model = Model(model.model)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), 
                                lr=lr, weight_decay=0.015)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.0007, 
        steps_per_epoch=len(train_dl), 
        epochs=epochs)

    beta = 1.0
    cutmix_prob = 0.5

    torch.cuda.empty_cache()

    model.to(device)

    running_losses = []

    for e in range(epochs):
        running_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0

        model.train()
        corr = 0
        
        tqdm_stream_train = tqdm(train_dl)
        for i, d in enumerate(tqdm_stream_train):
            X_batched = d[0]
            y_batched = d[1] #type(torch.float32)

            X_batched = X_batched.to(device) #cuda()
            y_batched = y_batched.to(device) # (N,), torch.int64

            optimizer.zero_grad()
            
            ##################################################
            # CUT-MIX: https://arxiv.org/abs/1905.04899
            r = np.random.rand(1)
            if beta > 0 and r < cutmix_prob:
                lam = np.random.beta(beta, beta)
                rand_idx = torch.randperm(X_batched.shape[0]).to(device)
                target_a = y_batched
                target_b = y_batched[rand_idx]
                bbx1, bby1, bbx2, bby2 = rand_bbox(X_batched.size(), lam)
                X_batched[:, :, bbx1:bbx2, bby1:bby2] = X_batched[rand_idx, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ( (bbx2-bbx1)*(bby2-bby1) / (X_batched.shape[-1]*X_batched.shape[-2]) )
                
                y_pred = model(X_batched)
                loss = criterion(y_pred, target_a) * lam + criterion(y_pred, target_b)*(1.-lam)
            ##################################################
            else:
                y_pred = model(X_batched) #.squeeze() #(N,1)->(N,), torch.float32
                loss = criterion(y_pred, y_batched) #(N,100), (N,)
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            corr += (torch.argmax(y_pred,dim=-1) == y_batched).sum()

            tqdm_stream_train.set_description(f"Trainning Epoch{e+1:3d}")
            
            scheduler.step()
        
        running_loss = running_loss / len(train_dl)
        running_losses.append(running_loss)
        train_acc = corr / len(train_ds)

        print(f"RESULT Epoch{e+1:3d}, Train Loss:{running_loss:.6f}, Train Acc.:{train_acc:.6f}") #, Valid Acc.:{val_acc:.6f}")
    
    #add dump model
    return model

if __name__ == '__main__':
    main()
