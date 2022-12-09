import torch
import torchvision
from PIL import Image
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from models.autoencoder import AutoencoderKL

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

path = './lightning_logs/version_7/checkpoints/epoch=28-step=29000.ckpt'

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

test_dataset = CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)

eval_loader = torch.utils.data.DataLoader(test_dataset,
                                batch_size=64,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=True,
                                drop_last = True)

if __name__ == '__main__':
    model = AutoencoderKL(ddconfig=ddconfig, embed_dim=ddconfig["z_channels"], ckpt_path=path)
    
    for batch_idx, (x,y) in enumerate(eval_loader):
        if batch_idx % 100 == 0:
            log = model.log_images(x)
            root = os.path.join("./samples", "vae_eval")
            for k in log:
                grid = torchvision.utils.make_grid(log[k], nrow=4)
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = "{}_{}_{}.png".format(
                    k,
                    "eval",
                    batch_idx)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(grid).save(path)

