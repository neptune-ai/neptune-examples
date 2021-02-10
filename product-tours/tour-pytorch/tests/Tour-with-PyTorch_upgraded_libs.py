# Tour with PyTorch

# Install dependencies

get_ipython().system(' pip install --quiet torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html scikit-learn==0.24.1 neptune-client==0.5.0')

get_ipython().system(' pip install --quiet --upgrade torch torchvision -f https://download.pytorch.org/whl/torch_stable.html scikit-learn neptune-client')

# Introduction

# Logging PyTorch meta-data to Neptune

## Basic example

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

### Initialize Neptune

import neptune

neptune.init('shared/tour-with-pytorch', api_token='ANONYMOUS')

### Create an experiment and log model hyper-parameters

neptune.create_experiment(name='pytorch-run',
                          tags=['pytorch', 'MNIST'],
                          params=PARAMS)

### Log data version to the experiment

dataset = datasets.MNIST('../data',
                         train=True,
                         download=True,
                         transform=transforms.Compose([transforms.ToTensor()]))

neptune.set_property('data_version',
                     hashlib.md5(dataset.data.cpu().detach().numpy()).hexdigest())

### Log losses, accuracy score and image predictions during training

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

# tests
exp = neptune.get_experiment()

### Stop Neptune experiment after training

# tests
# check logs
correct_logs_set = {'batch_loss', 'batch_acc', 'predictions'}
from_exp_logs = set(exp.get_logs().keys())

assert correct_logs_set == from_exp_logs, '{} - incorrect logs'.format(exp)

# check parameters
assert set(exp.get_parameters().keys()) == set(PARAMS.keys()), '{} parameters do not match'.format(exp)

neptune.stop()

## Summary

# If you want to learn more, go to the [Neptune documentation](https://docs.neptune.ai/integrations/pytorch.html).