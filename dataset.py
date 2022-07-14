"""
Custom dataloader for GTSRB downloaded from kaggle
note: GTSRB is also available from torchvision
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
import numpy as np

n_classes = 43

base_transforms = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

augmentation_transforms = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.ColorJitter(hue=0.05, saturation=0.1),
    T.RandomAffine(degrees=(-10, 10), translate=(0.2, 0.2), scale=(0.8, 1.1)),
    T.RandomPerspective(distortion_scale=0.2, p=0.5),
    T.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])


def get_dataset_balance():
    train_annotations = pd.read_csv(Path(Path(__file__).parent.absolute(), "data", "Train.csv"))
    plt.hist(train_annotations['ClassId'].to_numpy(), n_classes, (0, n_classes))
    plt.show()

# split to train and val by class
def split_train_val(ratio=0.2):
    train_annotations = pd.read_csv(Path(Path(__file__).parent.absolute(), "data", "Train.csv"))
    train = val = train_annotations[0:0] # empty except  headers
    for i in range(n_classes):
        classonly = train_annotations[train_annotations['ClassId'] == i] 
        train_i, val_i = train_test_split(classonly, test_size=ratio)
        train = pd.concat([train, train_i])
        val = pd.concat([val, val_i])
    return train, val


def get_test():
    test = pd.read_csv(Path(Path(__file__).parent.absolute(), "data", "Test.csv"))
    return test


# A custom Dataset class must implement three functions: __init__, __len__, and __getitem__
class GTSRB(Dataset):
    def __init__(self, data, use_augmentation=False):
        self.data = data
        self.transform  = augmentation_transforms if use_augmentation else base_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
                idx = idx.tolist()

        img_path = self.data.iloc[idx]["Path"]
        img_path = Path(Path(__file__).parent.absolute(),"data", img_path)
        img = Image.open(img_path)
        classId = self.data.iloc[idx]["ClassId"]

        if self.transform is not None:
            img = self.transform(img)

        return img, classId


if __name__ == "__main__":
    # get_dataset_balance()
    train, val = split_train_val(ratio=0.2)
    # train_dataset = GTSRB(train, use_augmentation=True)
    val_dataset = GTSRB(val, use_augmentation=True)
    print(val_dataset[2229])

