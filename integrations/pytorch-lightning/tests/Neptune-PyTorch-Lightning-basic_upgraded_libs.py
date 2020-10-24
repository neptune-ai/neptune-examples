# PyTorch Lightning 1.x + Neptune [Basic Example]

# Before you start

## Install dependencies

get_ipython().system(' pip install --user pytorch-lightning==1.0.0 neptune-client==0.4.123 torch==1.6.0 torchvision==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html')

get_ipython().system(' pip install --user --upgrade pytorch-lightning neptune-client torch torchvision')

# Step 1: Import Libraries

import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl

# Step 2: Define Hyper-Parameters

PARAMS = {'max_epochs': 3,
          'learning_rate': 0.005,
          'batch_size': 32}

# Step 3: Define LightningModule and DataLoader

# pl.LightningModule
class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=PARAMS['learning_rate'])

# DataLoader
train_loader = DataLoader(MNIST(os.getcwd(), download=True, transform=transforms.ToTensor()),
                          batch_size=PARAMS['batch_size'])

# Step 4: Create NeptuneLogger

from pytorch_lightning.loggers.neptune import NeptuneLogger

neptune_logger = NeptuneLogger(
    api_key="ANONYMOUS",
    project_name="shared/pytorch-lightning-integration",
    params=PARAMS)

# Step 5: Pass NeptuneLogger to the Trainer

trainer = pl.Trainer(max_epochs=PARAMS['max_epochs'],
                     logger=neptune_logger)

# Step 6: Run experiment

model = LitModel()

trainer.fit(model, train_loader)

# Explore Results

# tests
exp = neptune_logger.experiment.id

## check dataloader size
if len(train_loader) != 1875:
    raise ValueError('data loader size does not match')

## check logs
correct_logs = ['train_loss', 'epoch']

if set(neptune_logger.experiment.get_logs().keys()) != set(correct_logs):
    raise ValueError('incorrect metrics')