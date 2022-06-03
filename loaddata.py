import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from data.aircraft import Aircraft

IMG_SIZE = 299

class Dataset():
    def __init__(self, basedir) -> None:
        self.basedir = basedir
        self.train_ds = self.transform_data(train=True)
        self.test_ds = self.transform_data()
        self.image_ids, self.targets, self.classes, self.class_to_idx = self.train_ds.find_classes()
        self.classes = [c[:-1] for c in self.classes]
        self.class_to_idx = {c[:-1]:idx for c, idx in self.class_to_idx.items()}
    def transform_data(self, train=False):
        if train :
            transform = T.Compose(
                            [
                                T.Resize((IMG_SIZE,IMG_SIZE)),
                                T.ToTensor()
                            ]
                        )
        else:
            transform = T.Compose(
                            [
                                T.Resize((IMG_SIZE,IMG_SIZE)),
                                T.RandomHorizontalFlip(p=0.5),
                                T.RandomVerticalFlip(p=0.3),
                                T.RandomRotation(degrees=(-15, 15)),
                                T.ToTensor()
                            ]
                        )
        data = Aircraft(self.basedir + '/data/aircraft', train=train, download=False, transform=transform)
        return data

    def dataloader(self, batch_size, train=False):
        if train:
            dataset = self.train_ds
        else:
            dataset = self.test_ds

        data_loader = DataLoader(dataset, batch_size=batch_size,   
                                shuffle=True, num_workers=2, pin_memory=True)
        return data_loader

