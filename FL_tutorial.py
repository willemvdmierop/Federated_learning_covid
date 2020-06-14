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
        self.epochs = 1
        self.lr = 0.001
        self.seed = 1
        self.log_interval = 10
        self.precision_fractional = 3
        self.create_dataset = True
        self.save_model = True
        self.stage = 'test'


args = Arguments()
torch.manual_seed(args.seed)
image_size = [64, 64]
Features = image_size[0]
batch_size = 10
nc = 3
# ===================================================== create dataset =================================================
# covid has label 1 others has label 0
data_args = {'data_dir': './final_dataset', 'stage': args.stage, 'create_dataset': True, 'img_size': image_size}
dataset = dataset.FL_data(**data_args)
train_data_params = {'batch_size': batch_size, 'shuffle': True, 'pin_memory': True}
train_dataloader = data.DataLoader(dataset, **train_data_params)
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

model_pars = {'in_planes': Features, 'channels': nc, 'batch_size': batch_size}
model_ResNet = ResNet_Model.ResNet(ResNet_Model.BasicBlock, [2, 2, 2, 2], **model_pars)
model_ResNet = model_ResNet.to(device)
optimizer_ResNet = optim.SGD(model_ResNet.parameters(), lr=args.lr)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(13, 32)
        self.fc2 = nn.Linear(32, 24)
        self.fc3 = nn.Linear(24, 1)

    def forward(self, x):
        x = x.view(-1, 3*64*64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()
optimizer = optim.SGD(model.parameters(), lr=args.lr)
# print('model', model_ResNet)
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


# Simple FL_training without secure aggregation
def build_distributed_dataset():
    train_distributed_dataset = []
    for batch_idx, (data, target) in enumerate(train_dataloader):
        # Send data to the appropriate worker
        data = data.send(workers[batch_idx % len(workers)])
        # Send target to the appropriate worker
        target = target.send(workers[batch_idx % len(workers)])
        # add to dataset
        train_distributed_dataset.append((data, target))
    return train_distributed_dataset


train_distributed_dataset = build_distributed_dataset()
#print(train_distributed_dataset)

loss_list = []
def train_ResNet(epoch):
    model_ResNet.train()
    for batch_idx, (data, target) in enumerate(train_distributed_dataset):
        worker = data.location
        model_ResNet.send(worker)
        #model.send(worker)
        optimizer_ResNet.zero_grad()

        pred = model_ResNet(data)
        #pred = model(data)
        loss = F.mse_loss(pred.view(-1), target.type(torch.FloatTensor))
        loss.backward()
        optimizer_ResNet.step()
        #optimizer.step()
        #model.get()
        model_ResNet.get()

        if batch_idx % args.log_interval == 0:
            loss = loss.get()
            loss_list.append(loss.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_dataloader),
                       100. * batch_idx / len(train_dataloader), loss.item()))


start_time = time.time()
for epoch in range(1, args.epochs + 1):
    train_ResNet(epoch)
    if (args.save_model):
        torch.save(model_ResNet.state_dict(), os.path.join(dirname,'Resnet_covid_FL.pth'))
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer_ResNet.state_dict(),
            'loss_list': loss_list,
        }, os.path.join(dirname,'checkpoint.pth'))
        loss_list_array = np.array(loss_list)
        df = pd.DataFrame(loss_list_array, columns = ['Loss'])
        df.to_csv(os.path.join(dirname,"FL_covid_train.csv"))


total_time = time.time() - start_time
print('Total', round(total_time, 2), 's')

#epochs = np.arange(1,len(loss_list))

#fig, ax = plt.subplots(1, 1, figsize=(12, 4))
#ax.plot(epochs, loss_list, label ='loss')