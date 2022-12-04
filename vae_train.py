## Modules ##
import torch
import pytorch_lightning as pl
from models.autoencoder import AutoencoderKL
from tqdm import tqdm

## Datasets ##
import torchvision.transforms as transforms
from datasets.FashionMNIST import FashionMNIST

## Parser ##
from argparse import ArgumentParser

def main(hparams, ddconfig):
    model = AutoencoderKL(ddconfig=ddconfig, embed_dim=64)
    trainer = pl.Trainer(accelerator=hparams.accelerator, devices=hparams.devices, max_epochs=20)
    
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
    parser.add_argument("--batch_size", default=64)
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
