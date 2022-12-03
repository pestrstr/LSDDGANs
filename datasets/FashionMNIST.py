import os
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset

# FashionMNIST reader
def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def mnist_train(path):
    X_train, y_train = load_mnist(path, kind='train')
    return X_train, y_train

def mnist_test(path):
    X_test, y_test = load_mnist(path, kind='t10k')
    return X_test, y_test


class FashionMNIST(Dataset):
    def __init__(self, path, transform, train=True):
        super().__init__
        self.train = train
        if self.train is True:
            self.train_data, self.train_label = mnist_train(path)    
            self.train_data = self.train_data.reshape((-1, 1, 28, 28))
            self.train_data = np.transpose(self.train_data, (0, 2, 3, 1))
        else:
            self.test_data, self.test_label = mnist_train(path)
            self.test_data = self.test_data.reshape((-1, 1, 28, 28))
            self.train_data = np.transpose(self.train_data, (0, 2, 3, 1)) 
        self.transform = transform

    def __len__(self):
        if self.train is True:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, idx):
        if self.train is True:
            return self.transform(self.train_data[idx]), self.train_label[idx]
        else:
            return self.transform(self.test_data[idx]), self.test_label[idx]