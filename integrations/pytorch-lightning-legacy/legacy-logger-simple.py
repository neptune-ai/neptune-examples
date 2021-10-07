# README
# This is legacy-logger example
# User guide: https://docs-legacy.neptune.ai/integrations/pytorch_lightning.html
# Docstrings: https://docs-legacy.neptune.ai/api-reference/neptunecontrib/monitoring/pytorch_lightning/index.html

# For new logger go here: https://docs.neptune.ai/integrations-and-supported-tools/model-training/pytorch-lightning

# requirements:
# pip3 install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# pip install pytorch-lightning==1.4.8 neptune-client==0.9.8
# pip install 'neptune-contrib[monitoring]'

# Step 1: Import Libraries
import os

import pytorch_lightning as pl
import torch
from neptunecontrib.monitoring.pytorch_lightning import NeptuneLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

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
train_loader = DataLoader(
    MNIST(os.getcwd(), download=True, transform=transforms.ToTensor()),
    batch_size=PARAMS['batch_size'],
)

# Step 4: Create NeptuneLogger
neptune_logger = NeptuneLogger(
    project_name="shared/pytorch-lightning-integration",
    params=PARAMS,
    experiment_name="legacy-logger",
    tags=["legacy-logger"],
)

# Step 5: Pass NeptuneLogger to the Trainer
trainer = pl.Trainer(
    max_epochs=PARAMS['max_epochs'],
    logger=neptune_logger
)

# Step 6: Run experiment
model = LitModel()
trainer.fit(model, train_loader)

# Step 7: Stop Neptune logger at the end
neptune_logger.experiment.stop()
