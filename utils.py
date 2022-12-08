import sys
import os
import glob
import torch
from models.autoencoder import AutoencoderKL

# TODO: Add other configuration
config_dict = {
    (4, 3): {
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
    },
    (8, 4): { 
      "double_z": True,
      "z_channels": 4,
      "resolution": 32,
      "in_channels": 3,
      "out_ch": 3,
      "ch": 128,
      "ch_mult": [1,2,4,4],  
      "num_res_blocks": 2,
      "attn_resolutions": [],
      "dropout": 0.0,
      "lr": 0.001
    }
}


# Load trained VAE model
def load_VAE_Model(device, f=4, d=3, embed_dim=3, ppath="./lightning_logs"):
    model_path = "{path}/f={a}_d={b}_embed_dim={c}/checkpoints".format(path=ppath, a=f, b=d, c=embed_dim)
    checkpoints = glob.glob(f'{model_path}/*.ckpt')
    print(f"found {checkpoints}")
    if len(checkpoints) == 0:
        print("No VAE checkpoints found for setup f={a}, d={b} at path {p}".format(a=f, b=d, p=model_path))
        sys.exit(0)
    # Only one checkpoint expected in the directory        
    checkpoint_file = checkpoints[0]
    model = AutoencoderKL(ddconfig=config_dict[(f,d)], embed_dim=embed_dim)
    model_dict = torch.load(checkpoint_file, map_location=device)
    # print(model_dict)
    model.load_state_dict(model_dict["state_dict"])
    model.to(device)
    return model

def load_pretrained(device, f=8, d=4, type='kl'):
  model_path = f'pretrained/{type}-f{f}'
  checkpoints = glob.glob(f'{model_path}/*.ckpt')
  print(f"found {checkpoints}")
  if len(checkpoints) == 0:
      print("No VAE checkpoints found for setup f={a}, d={b} at path {p}".format(a=f, b=d, p=model_path))
      sys.exit(0)
  checkpoint_file = checkpoints[0]
  model = AutoencoderKL(ddconfig=config_dict[(f,d)], embed_dim=d)
  model_dict = torch.load(checkpoint_file, map_location=device)
  # print(model_dict)
  model.load_state_dict(model_dict["state_dict"])
  model.to(device)
  return model