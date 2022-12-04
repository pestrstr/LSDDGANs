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
        root = os.path.join(save_dir, "lightning_logs/images", split)
        for k, img in enumerate(images):
            grid = torchvision.utils.make_grid(img, nrow=4)
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
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, **self.log_images_kwargs)["samples"]

            N = min(images.shape[0], self.max_images)
            images = images[:N]
            if isinstance(images, torch.Tensor):
                images = images.detach().cpu()
                if self.clamp:
                    images = torch.clamp(images, -1., 1.)

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
            self.log_img(pl_module, batch, batch_idx, split="train")
    # hook to call at every val batch end
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")


def main(hparams, ddconfig):
    model = AutoencoderKL(ddconfig=ddconfig, embed_dim=64)
    logger_callback = ImageLogger(batch_frequency=200, max_images=1)
    trainer = pl.Trainer(accelerator=hparams.accelerator, 
                          devices=hparams.devices, 
                          max_epochs=hparams.max_epochs,
                          callbacks=[logger_callback])
    
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
    parser.add_argument("--batch_size", default=100)
    parser.add_argument("--max_epochs", default=20)
    args = parser.parse_args()

    main(args, ddconfig)

## Testing Function ##
class PerceptualVAETrainer():
  def __init__(self, config):
    super().__init__()
    self.model = config["model"]
    self.optimizer_AE = config["optimizer_AE"]
    self.optimizer_D = config["optimizer_D"]
    self.train_loader = config["train_loader"]
    self.test_loader = config["test_loader"]
    self.n_epochs = config["n_epochs"]
    self.batch_size = config["batch_size"]

  def train(self):
    self.model.train()
    for epoch in range(self.n_epochs):

        #TQDM Setting
        tq = tqdm(total = len(self.train_loader) * self.batch_size, position=0, leave=True) 
        tq.set_description('epoch %d' % (epoch))

        for _, (x,y) in enumerate(self.train_loader):
          # Train Encoder+Decoder+logvar
          self.optimizer_AE.zero_grad()
          loss_AE = self.model.training_step(x, None, 0)
          loss_AE.backward()
          self.optimizer_AE.step()

          # Train Discriminator
          self.optimizer_D.zero_grad()
          loss_D = self.model.training_step(x, None, 1)
          loss_D.backward()
          self.optimizer_D.step()

          #Print statistics
          tq.update(self.batch_size)
          tq.set_postfix({"AE Loss" : f'{loss_AE.item():.6f}', "D Loss" : f'{loss_D.item():.6f}'})    
    
    tq.close()
  
  def evaluate(self):
    # TODO: Implement evaluate function
    pass
