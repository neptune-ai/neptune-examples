# PyTorch Lightning + Neptune [Advanced Example]
# 
# Neptune helps you keep track of your machine learning experiments and if you are using PyTorch Lightning you can add tracking very easily. 
# 
# Let me show you how.

## Install dependencies 

# Not all of those are a must but I wanted to show more cool stuff.

# ! pip install pytorch-lightning scikit-plot neptune-client neptune-contrib[viz] --upgrade

# Basic imports

import os

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl

# Define parameters

MAX_EPOCHS=7
LR=0.02
BATCHSIZE=32
CHECKPOINTS_DIR = 'my_models/checkpoints/'

# Define LightningModule

# This is your typical `pl.LightningModule` with required methods defined. Nothing new here.

class AdvancedSystem(pl.LightningModule):

    def __init__(self):
        super(BasicSystem, self).__init__()
        # not the best model...
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))
    
    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        fig = plt.figure()
        losses = np.stack([x['val_loss'].numpy() for x in outputs])
        plt.hist(losses)
        self.logger.experiment.log_image('loss_histograms', fig)
        plt.close(fig)

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=LR)
    
    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=BATCHSIZE)
    
    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=BATCHSIZE)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=BATCHSIZE)

# Define NeptuneLogger with custom params

We will also create a more advanced `NeptuneLogger` that contains information about hyperparameters, add tags to make runs organized, and define which scripts we want to snapshot.

Also, I will use the `close_after_fit=False` argument to make sure that the logger doesn't close after the `.fit` loop ends.

from pytorch_lightning.loggers.neptune import NeptuneLogger

neptune_logger = NeptuneLogger(
    api_key="ANONYMOUS",
    project_name="shared/pytorch-lightning-integration",
    close_after_fit=False,
    experiment_name="default",  # Optional,
    params={"max_epochs": MAX_EPOCHS,
            "batch_size": BATCHSIZE,
            "lr": LR}, # Optional,
    tags=["pytorch-lightning", "mlp"],
    upload_source_files=['*.py','*.yaml'],
)

# Pass neptune_logger to the Trainer

Again we need to pass `neptune_logger` to the `Trainer` object:

from pytorch_lightning import Trainer

model_checkpoint = pl.callbacks.ModelCheckpoint(filepath=CHECKPOINTS_DIR)

advanced_model = AdvancedSystem()
trainer = Trainer(max_epochs=MAX_EPOCHS,
                  logger=neptune_logger,
                  checkpoint_callback=model_checkpoint,
                  )
trainer.fit(advanced_model)

# and we can explore our run in Neptune.
# 
# But since we specified `close_after_fit=False` we can log additional things like:

## Test metrics from `.test(...)` call 

trainer.test(advanced_model)

## Custom metrics 

We can log metrics that we want to calculate after .fit ends.
For example let's calculate `accuracy_score` and use `.log_metric` method to log it to Neptune.

advanced_model.freeze()
test_loader = DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=256)

y_true, y_pred = [],[]
for i, (x, y) in enumerate(test_loader):
    y_hat = advanced_model.forward(x).argmax(axis=1).cpu().detach().numpy()
    y = y.cpu().detach().numpy()

    y_true.append(y)
    y_pred.append(y_hat)

    if i == len(test_loader):
        break
y_true = np.hstack(y_true)
y_pred = np.hstack(y_pred)

# Log additional metrics
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
neptune_logger.experiment.log_metric('test_accuracy', accuracy)

## Performance charts

You can log performance charts like ROC AUC or Confusion Matrix.

Just use `.log_image` method on a matplotlib figure you want to log.

from scikitplot.metrics import plot_confusion_matrix

fig, ax = plt.subplots(figsize=(16, 12))
plot_confusion_matrix(y_true, y_pred, ax=ax)
neptune_logger.experiment.log_image('confusion_matrix', fig)

# Log artifacts

You can log any file to Neptune. just use the `.log_artifact` method.

For example, we can log the entire 'CHECKPOINTS_DIR' directory.

neptune_logger.experiment.log_artifact(CHECKPOINTS_DIR)

# Stop the logger

# After everything is done you need to stop the logger.

neptune_logger.experiment.stop()

# Explore in Neptune

# Now you can explore everything you logged in Neptune.

# You can use your experiment link or go check out [this experiment](https://ui.neptune.ai/o/shared/org/pytorch-lightning-integration/e/PYTOR-119/logs):

# ![image](https://neptune.ai/wp-content/uploads/lightning_advanced.gif)

# Fetch experiments after training
# 
# Neptune lets you access everything you logged programatically.

# Fetch experiment dashboard

# You can get the dashboard table into `pandas.DataFrame`

import neptune

project = neptune.init(api_token="ANONYMOUS",
                       project_qualified_name='shared/pytorch-lightning-integration')
project.get_leaderboard().head(3)

# Visualize experiments with Hiplot

# With [Neptune - HiPlot integration](https://docs.neptune.ai/integrations/hiplot.html) you can visualize all your experiment metrics and hyperparameters. 

from neptunecontrib.viz.parallel_coordinates_plot import make_parallel_coordinates_plot

make_parallel_coordinates_plot(metrics= ['train_loss', 'val_loss', 'test_accuracy'],
                               params = ['max_epochs', 'batch_size', 'lr'])

# Update Experiment

# You can also fetch a single experiment and update it with some external metric calculated after training.

exp = project.get_experiments(id='PYTOR-63')[0]
exp.log_metric('some_external_metric', 0.92)

# Create your free account
# 
# The best part is, Neptune is completely free for individuals and research teams so you can go ahead and [create your free account](https://neptune.ai?utm_source=colab&utm_medium=notebook&utm_campaign=integration-pytorch-lightning) and check it out for yourself.