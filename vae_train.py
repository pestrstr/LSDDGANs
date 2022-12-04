## Modules ##
import torch
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
    
    fmnist_train_loader = torch.utils.data.DataLoader(fmnist_train,
                                            batch_size=hparams.batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            pin_memory=True,
                                            drop_last = True)

    fmnist_val_loader = torch.utils.data.DataLoader(fmnist_val,
                                            batch_size=hparams.batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            pin_memory=True,
                                            drop_last = True)

    trainer.fit(model, fmnist_train_loader, fmnist_val_loader)


if __name__ == '__main__':
    ddconfig = {
      "double_z": True,
      "z_channels": 64,
      "resolution": 256,
      "in_channels": 1,
      "out_ch": 1,
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
    parser.add_argument("--batch-size", default=64)
    args = parser.parse_args()

    main(args, ddconfig)
