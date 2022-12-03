## Modules ##
import pytorch_lightning as pl
from models.autoencoder import AutoencoderKL

## Datasets ##
import torchvision.transforms as transforms
from datasets.FashionMNIST import FashionMNIST

## Parser ##
from argparse import ArgumentParser

def main(hparams, ddconfig):
    model = AutoencoderKL(ddconfig=ddconfig, embed_dim=64)
    trainer = pl.Trainer(accelerator=hparams.accelerator, devices=hparams.devices)
    transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))])
    fmnist_train = FashionMNIST(path='./datasets/fashion_mnist', transform=transform)
    fmnist_val = FashionMNIST(path='./datasets/fashion_mnist', transform=transform, train=False)
    trainer.fit(model, fmnist_train, fmnist_val)


if __name__ == '__main__':
    ddconfig = {
      "double_z": True,
      "z_channels": 64,
      "resolution": 256,
      "in_channels": 3,
      "out_ch": 3,
      "ch": 128,
      "ch_mult": [1,1,2,2,4,4],  
      "num_res_blocks": 2,
      "attn_resolutions": [16,8],
      "dropout": 0.0,
      "lr": 0.001
    }
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--devices", default=1)
    args = parser.parse_args()

    main(args, ddconfig)
