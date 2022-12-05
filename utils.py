import sys
import os
import glob
import torch

# Load trained VAE model
def load_VAE_Model(f=4, d=3, ppath="./lightning_logs", device):
    model_path = "{path}/f={a}_d={b}/checkpoints".format(path=ppath, a=f, b=d)
    checkpoints = glob.glob(f'{model_path}/*.ckpt')
    if len(checkpoints) == 0:
        print("No VAE checkpoints found for setup f={a}, d={b}".format(a=f, b=d))
        sys.exit(0)
    # Only one checkpoint expected in the directory        
    checkpoint_file = checkpoints[0]
    if device is None:
        device = 'cpu'
    return torch.load(checkpoint_file, map_location=device)


