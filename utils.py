import sys
import os

def load_VAE_Model(f=4, d=3, ppath="./lightning_logs"):
    model_path = "{path}/f={a}_d={b}/checkpoints".format(path=ppath, a=f, b=d)
    

