# Neptune + PyTorch

# Before we start

## Install dependencies

get_ipython().system(' pip install --quiet -f https://download.pytorch.org/whl/torch_stable.html torch==1.7.0 neptune-client==0.4.126')

## Import libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

## Define your model, data loaders and optimizer

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data',
                   train=True,
                   download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),
   batch_size=64,
   shuffle=True)

model = Net()

optimizer = optim.SGD(model.parameters(), 0.005, 0.9)

## Initialize Neptune

import neptune

neptune.init(api_token='ANONYMOUS', project_qualified_name='shared/pytorch-integration')

# Quickstart

## Step 1: Create an Experiment

neptune.create_experiment('pytorch-quickstart')

## Step 2: Add logging into your training loop

for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    outputs = model(data)
    loss = F.nll_loss(outputs, target)

    # log loss
    neptune.log_metric('batch_loss', loss)

    loss.backward()
    optimizer.step()
    if batch_idx == 100:
        break

# Step 3: Explore results in the Neptune UI

## Step 4: Stop logging

# tests

exp = neptune.get_experiment()
all_logs = exp.get_logs()

## check logs
correct_logs = ['batch_loss']

if set(all_logs.keys()) != set(correct_logs):
    raise ValueError('incorrect metrics')

neptune.stop()

# Advanced Options

## Log hardware consumption

get_ipython().system(' pip install --quiet psutil==5.6.6')

## Log hyperparameters

PARAMS = {'lr':0.005, 
           'momentum':0.9, 
           'iterations':100}

optimizer = optim.SGD(model.parameters(), PARAMS['lr'], PARAMS['momentum'])

# log params
neptune.create_experiment('pytorch-advanced', params=PARAMS)

## Log image predictions

for batch_idx, (data, target) in enumerate(train_loader):
                              
    optimizer.zero_grad()
    outputs = model(data)
    loss = F.nll_loss(outputs, target)

    loss.backward()
    optimizer.step()
                              
    # log loss
    neptune.log_metric('batch_loss', loss)

    # log predicted images
    if batch_idx % 50 == 1:
        for image, prediction in zip(data, outputs):
            description = '\n'.join(['class {}: {}'.format(i, pred) 
                                     for i, pred in enumerate(F.softmax(prediction))])
            neptune.log_image('predictions', 
                              image.squeeze().detach().numpy() * 255, 
                              description=description)
                                               
    if batch_idx == PARAMS['iterations']:
        break

## Log model weights

torch.save(model.state_dict(), 'model_dict.ckpt')

# log model
neptune.log_artifact('model_dict.ckpt')

# Explore results in the Neptune UI

# tests

exp = neptune.get_experiment()
all_logs = exp.get_logs()

## check logs
correct_logs = ['batch_loss', 'predictions']

if set(all_logs.keys()) != set(correct_logs):
    raise ValueError('incorrect metrics')

## Stop logging

neptune.stop()