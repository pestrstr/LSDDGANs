## Modules ##
import torch
import torchvision
from PIL import Image
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from models.autoencoder import AutoencoderKL
from tqdm import tqdm

## Datasets ##
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from datasets.FashionMNIST import FashionMNIST

## Parser ##
from argparse import ArgumentParser

# see PyTorch Lightning Callbacks. This callback is automatically called by PyTorch Lightnining.
# Based on the hooks that I've overloaded, the callback will be called every batch_freq batches

class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
            hasattr(pl_module, "log_images") and
              callable(pl_module.log_images) and
              self.max_images > 0):

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            # Testing reconstruction capabilities
            inputs = batch[0][:self.max_images]

            with torch.no_grad():
                images = pl_module.log_images(inputs, **self.log_images_kwargs)
                      
            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)
            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0) and (
                check_idx > 0 or self.log_first_step):
            return True
        return False

    # hook to call at every train batch end
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")

def main(hparams, ddconfig):
    model = AutoencoderKL(ddconfig=ddconfig, embed_dim=hparams.embed_dim)
    logger_callback = ImageLogger(batch_frequency=200, max_images=64)
    trainer = pl.Trainer(accelerator=hparams.accelerator, 
                          devices=hparams.devices, 
                          max_epochs=hparams.max_epochs,
                          callbacks=[logger_callback])

    if hparams.dataset == 'cifar10':

        train_dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True) 
        test_dataset = CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=hparams.batch_size,
                                        shuffle=True,
                                        num_workers=4,
                                        pin_memory=True,
                                        drop_last = True)

        eval_loader = torch.utils.data.DataLoader(test_dataset,
                                        batch_size=hparams.batch_size,
                                        shuffle=False,
                                        num_workers=4,
                                        pin_memory=True,
                                        drop_last = True)


    elif hparams.dataset == 'fashion_mnist':
        transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))])
        fmnist_train = FashionMNIST(path='./datasets/fashion_mnist', transform=transform)
        fmnist_val = FashionMNIST(path='./datasets/fashion_mnist', transform=transform, train=False)
    
        train_loader = torch.utils.data.DataLoader(fmnist_train,
                                            batch_size=hparams.batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            pin_memory=True,
                                            drop_last = True)

        eval_loader = torch.utils.data.DataLoader(fmnist_val,
                                            batch_size=hparams.batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            pin_memory=True,
                                            drop_last = True)

    trainer.fit(model, train_loader, eval_loader)


if __name__ == '__main__':
  # f = 4, d = 3, z 3x8x8
    ddconfig = {
      "double_z": True,
      "z_channels": 3,
      "resolution": 32,
      "in_channels": 3,
      "out_ch": 3,
      "ch": 128,
      "ch_mult": [1,2,4],  
      "num_res_blocks": 2,
      "attn_resolutions": [],
      "dropout": 0.0,
      "lr": 0.001
    }

    parser = ArgumentParser()
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--devices", default=1)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--embed_dim", type=int, default=ddconfig["z_channels"])
    parser.add_argument("--dataset", type=str, default='cifar10')
    args = parser.parse_args()

    main(args, ddconfig)
