import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import numpy as np
import os
import dataset
import ResNet_Model
import syft as sy


class Arguments():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 64
        self.epochs = 10
        self.lr = 0.02
        self.seed = 1
        self.log_interval = 1
        self.precision_fractional = 3
        self.create_dataset = True



args = Arguments()
torch.manual_seed(args.seed)

hook = sy.TorchHook(torch)


def connect_to_workers(n_workers):
    return [sy.VirtualWorker(hook, id=f"worker{i + 1}") for i in range(n_workers)]


def connect_to_crypto_provider():
    return sy.VirtualWorker(hook, id="crypto_provider")


workers = connect_to_workers(n_workers=4)
crypto_provider = connect_to_crypto_provider()

print("\nWorkers: {}".format(workers))
print("Crypto_provider", crypto_provider)
image_size = [64, 64]
Features = image_size[0]
nc = 3
# ===================================================== create dataset =================================================
# covid has label 1 others has label 0
data_args = {'data_dir': './final_dataset', 'stage': 'train', 'create_dataset': True, 'img_size': image_size}
dataset = dataset.FL_data(**data_args)

'''
# find something to balance this unbalanced dataset? solution is FL?
    ## COVID19 images are 138 in training-set
    ## OTHERS images are 5264 in training-set
    ## COVID19 images are 19 in test-set
    ## OTHERS images are 631 in test-set
    ## COVID19 images are 7 in validation-set
    ## OTHERS images are 19 in validation-set
'''

# ====================================================== create model ==================================================
dirname = 'FL_Resnet_covid_19'
wd = os.getcwd()
if not os.path.exists(os.path.join(wd, dirname)): os.mkdir(os.path.join(wd, dirname))

d_pars = {'in_planes': Features, 'channels': nc}

# ResNet18: parameters discriminator 11183318

model = ResNet_Model.ResNet(ResNet_Model.BasicBlock, [2, 2, 2, 2], **d_pars)

print('model', model)