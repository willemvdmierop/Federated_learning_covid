import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from PIL import Image as PILImage
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import dataset
import ResNet_Model
import syft as sy
import pandas as pd

device = torch.device('cpu')
if torch.cuda.is_available():
   device = torch.device('cuda')


class Arguments():
    def __init__(self):
        self.test_batch_size = 64
        self.batch_size = 10
        self.epochs = 1
        self.lr = 0.001
        self.seed = 1
        self.log_interval = 1
        self.precision_fractional = 3
        self.create_dataset = True
        self.save_model = True
        self.stage = 'val'


args = Arguments()
torch.manual_seed(args.seed)
image_size = [64, 64]
Features = image_size[0]
batch_size = 10
nc = 3
# ===================================================== create dataset =================================================
# covid has label 1 others has label 0
data_train_args = {'data_dir': './final_dataset', 'stage': 'train', 'create_dataset': True, 'img_size': image_size}
dataset_train = dataset.FL_data(**data_train_args)
train_data_params = {'batch_size': batch_size, 'shuffle': True, 'pin_memory': True}
train_dataloader = data.DataLoader(dataset_train, **train_data_params)

data_test_args = {'data_dir': './final_dataset', 'stage': 'test', 'create_dataset': True, 'img_size': image_size}
dataset_test = dataset.FL_data(**data_test_args)
test_data_params =  {'batch_size': batch_size, 'shuffle': True, 'pin_memory': True}
test_dataloader = data.DataLoader(dataset_test, **test_data_params)


'''
# find something to balance this unbalanced dataset? solution is FL?
    ## COVID19 images are 138 in training-set
    ## OTHERS images are 5264 in training-set
    ## COVID19 images are 19 in test-set
    ## OTHERS images are 631 in test-set
    ## COVID19 images are 7 in validation-set
    ## OTHERS images are 19 in validation-set
'''


# ====================================================== FL ==================================================

hook = sy.TorchHook(torch)


def connect_to_workers(n_workers):
    return [sy.VirtualWorker(hook, id=f"worker{i + 1}") for i in range(n_workers)]


def connect_to_crypto_provider():
    return sy.VirtualWorker(hook, id="crypto_provider")


workers = connect_to_workers(n_workers=4)
crypto_provider = connect_to_crypto_provider()

print("\nWorkers: {}".format(workers))
print("Crypto_provider", crypto_provider)

def get_private_data_loaders(precision_fractional, workers, crypto_provider):
    def secret_share(tensor):
        # transform to fixed precision and secret share a tensor
        return(tensor.fix_precision(precision_fractional=precision_fractional).share(*workers, crypto_provider = crypto_provider, requires_grad = True))

    private_train_loader = [(secret_share(data), secret_share(target)) for i, (data, target) in enumerate(train_dataloader)]

    private_test_loader = [(secret_share(data), secret_share(target)) for i, (data, target) in enumerate(test_dataloader)]

    return private_train_loader, private_test_loader

private_train_loader, private_test_loader = get_private_data_loaders(precision_fractional=args.precision_fractional,
                                                                     workers=workers,
                                                                     crypto_provider=crypto_provider)

print(private_train_loader)
print(private_test_loader)


def train(args, model, private_train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(private_train_loader):
        start_time = time.time()
        optimizer.zero_grad()
        predictions = model(data)
        batch_size = predictions.shape[0]
        loss = ((predictions - target)**2).sum().refresh()/batch_size
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            loss = loss.get().float_precision()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}s'.format(
                epoch, batch_idx * args.batch_size, len(private_train_loader) * args.batch_size,
                       100. * batch_idx / len(private_train_loader), loss.item(), time.time() - start_time))

def test(args, model, private_test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in private_test_loader:
            start_time = time.time()
            output = model(data)
            pred = output.argmax(dim = 1)
            correct += pred.eq(target.view_as(pred)).sum()

    correct = correct.get().float_precision()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct.item(), len(private_test_loader) * args.test_batch_size,
                        100. * correct.item() / (len(private_test_loader) * args.test_batch_size)))


# ====================================================== create model ==================================================
dirname = 'FL_Resnet_covid_19'
wd = os.getcwd()
if not os.path.exists(os.path.join(wd, dirname)): os.mkdir(os.path.join(wd, dirname))

model_pars = {'in_planes': Features, 'channels': nc, 'batch_size': batch_size}
model_ResNet = ResNet_Model.ResNet(ResNet_Model.BasicBlock, [2, 2, 2, 2], **model_pars)
model_ResNet = model_ResNet.fix_precision().share(*workers, crypto_provider = crypto_provider, requires_grad = True)

optimizer = optim.SGD(model_ResNet.parameters(), lr=args.lr)
optimizer = optimizer.fix_precision()

for epoch in range(1, args.epochs + 1):
    train(args, model_ResNet, private_train_loader, optimizer, epoch)
    test(args, model_ResNet, private_test_loader)
# print('model', model_ResNet)