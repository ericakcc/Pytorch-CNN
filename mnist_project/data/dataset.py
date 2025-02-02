import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MyDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        """
        df: DataFrame of shape (N, 785) if labeled, or (N, 784) if test.
        transform: torchvision transform for data augmentation/preprocessing.
        is_test: Whether this Dataset is for the test set (no labels).
        """
        self.is_test = is_test
        self.transform = transform

        if not self.is_test:
            self.labels = df['label'].values
            self.data = df.drop(columns=['label']).values.astype(np.uint8).reshape(-1, 28, 28)
        else:
            self.labels = None
            self.data = df.values.astype(np.uint8).reshape(-1, 28, 28)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform:
            img = self.transform(img)
        if self.is_test:
            return img
        else:
            return img, self.labels[idx]


def get_transforms(is_train=True, rotation=10):
    transform_list = [transforms.ToPILImage()]
    if is_train and rotation > 0:
        transform_list.append(transforms.RandomRotation(degrees=rotation))
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transforms.Compose(transform_list)