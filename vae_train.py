## Modules ##
import pytorch_lightning as pl
from models.autoencoder import AutoencoderKL

## Datasets ##
import torchvision.transforms as transforms
from datasets.FashionMNIST import FashionMNIST

if __name__ == '__main__':
    model = AutoencoderKL()
    trainer = pl.Trainer()
    transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))])
    fmnist_train = FashionMNIST(path='./datasets/fashion_mnist', transform=transform)
    fmnist_val = FashionMNIST(path='./datasets/fashion_mnist', transform=transform, train=False)
    trainer.fit(model, fmnist_train, fmnist_val)