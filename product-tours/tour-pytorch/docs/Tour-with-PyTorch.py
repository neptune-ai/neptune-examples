# Tour with PyTorch

# Install dependencies

# Introduction

# Logging PyTorch meta-data to Neptune

import hashlib

import neptune
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self, fc_out_features):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, fc_out_features)
        self.fc2 = nn.Linear(fc_out_features, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

PARAMS = {'fc_out_features': 400,
          'lr': 0.008,
          'momentum': 0.99,
          'iterations': 300,
          'batch_size': 64}

## Initialize Neptune

import neptune

neptune.init('shared/tour-with-pytorch', api_token='ANONYMOUS')

## Create an experiment and log model hyper-parameters

neptune.create_experiment(name='pytorch-run',
                          tags=['pytorch', 'MNIST'],
                          params=PARAMS)

## Log data version to the experiment

dataset = datasets.MNIST('../data',
                         train=True,
                         download=True,
                         transform=transforms.Compose([transforms.ToTensor()]))

neptune.set_property('data_version',
                     hashlib.md5(dataset.data.cpu().detach().numpy()).hexdigest())

## Log losses, accuracy score and image predictions during training

train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=PARAMS['batch_size'],
                                           shuffle=True)

model = Net(PARAMS['fc_out_features'])
optimizer = optim.SGD(model.parameters(), PARAMS['lr'], PARAMS['momentum'])

for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    outputs = model(data)
    loss = F.nll_loss(outputs, target)

    # Log loss
    neptune.log_metric('batch_loss', loss)

    y_true = target.cpu().detach().numpy()
    y_pred = outputs.argmax(axis=1).cpu().detach().numpy()
    acc = accuracy_score(y_true, y_pred)

    # Log accuracy
    neptune.log_metric('batch_acc', acc)

    loss.backward()
    optimizer.step()

    # Log image predictions
    if batch_idx % 50 == 1:
        for image, prediction in zip(data, outputs):
            description = '\n'.join(['class {}: {}'.format(i, pred)
                                     for i, pred in enumerate(F.softmax(prediction, dim=0))])
            neptune.log_image('predictions',
                              image.squeeze(),
                              description=description)

    if batch_idx == PARAMS['iterations']:
        break

## Log model weight to experiment

torch.save(model.state_dict(), 'model_dict.pth')
neptune.log_artifact('model_dict.pth')

## Stop Neptune experiment after training

neptune.stop()

# Summary

# If you want to learn more, go to the [Neptune documentation](https://docs.neptune.ai/integrations/pytorch.html).