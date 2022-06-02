import torch 
from loaddata import Dataset
from  tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model checkpoint Config', add_help=False)
    parser.add_argument("--modeldir", type=str, default='.',
                    help="Dir to the model's checkpoint")
    args = parser.parse_args()

    return args

def eval(model):
    dataset = Dataset()
    train_ds = dataset.train_ds
    test_ds = dataset.test_ds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval(); # efficientnet_b7 = models.efficientnet_b4()
    torch.cuda.empty_cache()

    train_dl2 = torch.utils.data.DataLoader(train_ds, batch_size=2,
                                        pin_memory=True)

    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=2,
                                        pin_memory=True)

    tqdm_stream_check_train = tqdm(train_dl2)
    corr = 0

    for j, d in enumerate(tqdm_stream_check_train):
        X_batched = d[0]
        y_batched = d[1]
        
        X_batched = X_batched.to(device) #cuda()
        y_batched = y_batched.to(device) #cuda()

        y_pred = model(X_batched)
        corr += (torch.argmax(y_pred,dim=-1) == y_batched).sum().item()

        tqdm_stream_check_train.set_description("Acc. for TRAIN SET")

    train_acc = corr / len(train_ds)
    print('Train acc:',train_acc)

    tqdm_stream_check_val = tqdm(test_dl)
    corr = 0

    for j, d in enumerate(tqdm_stream_check_val):
        X_batched = d[0]
        y_batched = d[1]
        
        X_batched = X_batched.to(device) #cuda()
        y_batched = y_batched.to(device) #cuda()

        y_pred = model(X_batched)
        corr += (torch.argmax(y_pred,dim=-1) == y_batched).sum().item()

        tqdm_stream_check_val.set_description("Acc. for TEST SET")

    test_acc = corr / len(test_ds)
    print('Test acc:',test_acc)

if __name__ == '__main__':
    args = parse_args()
    model = torch.load(args.modeldir)
    eval(model)