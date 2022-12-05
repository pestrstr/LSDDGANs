import sys
import os
import glob
import torch

# Load trained VAE model
def load_VAE_Model(f=4, d=3, ppath="./lightning_logs", device):
    model_path = "{path}/f={a}_d={b}/checkpoints".format(path=ppath, a=f, b=d)
    # Only one checkpoint expected in the directory
    checkpoint_file = glob.glob(f'{model_path}/*.ckpt')[0]
    if device is None:
        device = 'cpu'
    return torch.load(checkpoint_file, map_location=device)


