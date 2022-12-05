import sys
import os
import glob
import torch
from .models.autoencoder import AutoencoderKL

# TODO: Add other configuration
config_dict = {
    (4,3): {
      "double_z": True,
      "z_channels": 3,
      "resolution": 32,
      "in_channels": 1,
      "out_ch": 1,
      "ch": 128,
      "ch_mult": [1,2,4],  
      "num_res_blocks": 2,
      "attn_resolutions": [],
      "dropout": 0.0,
      "lr": 0.001
    },
    (32, 64): {
      "double_z": True,
      "z_channels": 64,
      "resolution": 32,
      "in_channels": 1,
      "out_ch": 1,
      "ch": 128,
      "ch_mult": [ 1,1,2,2,4,4],  # num_down = len(ch_mult)-1
      "num_res_blocks": 2,
      "attn_resolutions": [16,8],
      "dropout": 0.0,
      "lr": 0.001
    }
}


# Load trained VAE model
def load_VAE_Model(device, f=4, d=3, embed_dim=64, ppath="./lightning_logs"):
    model_path = "{path}/f={a}_d={b}/checkpoints".format(path=ppath, a=f, b=d)
    checkpoints = glob.glob(f'{model_path}/*.ckpt')
    if len(checkpoints) == 0:
        print("No VAE checkpoints found for setup f={a}, d={b}".format(a=f, b=d))
        sys.exit(0)
    # Only one checkpoint expected in the directory        
    checkpoint_file = checkpoints[0]
    if device is None:
        device = 'cpu'
    model = AutoencoderKL(ddconfig=config_dict[(f,d)], embed_dim=)
    return torch.load(checkpoint_file, map_location=device)


