import sys
import os
import glob

def load_VAE_Model(f=4, d=3, ppath="./lightning_logs"):
    model_path = "{path}/f={a}_d={b}/checkpoints".format(path=ppath, a=f, b=d)
    checkpoint = glob.glob(model_path)[0]
    print(checkpoint)

if __name__ == '__main__':
    ## Use this for testing
    load_VAE_Model()

