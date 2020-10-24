# PyTorch Lightning 1.x + Neptune [Advanced Example]

# Before you start

## Install necessary dependencies

get_ipython().system(' pip install --user pytorch-lightning==1.0.0 neptune-client==0.4.123 torch==1.6.0 torchvision==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html')

## Install additional dependencies

get_ipython().system(' pip install --user numpy==1.18.5 scikit-learn==0.23.2 scikit-plot==0.3.7')

# Step 1: Import Libraries

import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

# Step 2: Define Hyper-Parameters

LightningModule_Params = {'image_size': 28,
                          'linear': 128,
                          'n_classes': 10,
                          'learning_rate': 0.0023,
                          'decay_factor': 0.95}

LightningDataModule_Params = {'batch_size': 32,
                              'num_workers': 0,
                              'normalization_vector': ((0.1307,), (0.3081,)),}

LearningRateMonitor_Params = {'logging_interval': 'epoch'}

ModelCheckpoint_Params = {'filepath': 'my_model/checkpoints/{epoch:02d}-{val_loss:.2f}',
                          'save_weights_only': True,
                          'save_top_k': 3,
                          'monitor': 'val_loss',
                          'period': 1}

Trainer_Params = {'log_every_n_steps': 100,
                  'max_epochs': 7,
                  'track_grad_norm': 2}

ALL_PARAMS = {**LightningModule_Params,
              **LightningDataModule_Params,
              **LearningRateMonitor_Params,
              **ModelCheckpoint_Params,
              **Trainer_Params}

# Step 3: Define LightningModule, LightningDataModule and Callbacks

## 1: Implement LightningModule

class LitModel(pl.LightningModule):

    def __init__(self, image_size, linear, n_classes, learning_rate, decay_factor):
        super().__init__()
        self.image_size = image_size
        self.linear = linear
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.train_img_max = 10
        self.train_img = 0

        self.layer_1 = torch.nn.Linear(image_size * image_size, linear)
        self.layer_2 = torch.nn.Linear(linear, n_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = LambdaLR(optimizer, lambda epoch: self.decay_factor ** epoch)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=False)
        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()
        return {'loss': loss,
                'y_true': y_true,
                'y_pred': y_pred}

    def training_epoch_end(self, outputs):
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            y_true = np.append(y_true, results_dict['y_true'])
            y_pred = np.append(y_pred, results_dict['y_pred'])
        acc = accuracy_score(y_true, y_pred)
        self.log('train_acc', acc)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss, prog_bar=False)
        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()
        return {'loss': loss,
                'y_true': y_true,
                'y_pred': y_pred}

    def validation_epoch_end(self, outputs):
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            y_true = np.append(y_true, results_dict['y_true'])
            y_pred = np.append(y_pred, results_dict['y_pred'])
        acc = accuracy_score(y_true, y_pred)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss, prog_bar=False)
        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()
        for j in np.where(np.not_equal(y_true, y_pred))[0]:
            img = np.squeeze(x[j].cpu().detach().numpy())
            img[img < 0] = 0
            img = (img / img.max()) * 256
            neptune_logger.experiment.log_image('test_misclassified_images',
                                                img,
                                                description='y_pred={}, y_true={}'.format(y_pred[j], y_true[j]))
        return {'loss': loss,
                'y_true': y_true,
                'y_pred': y_pred}

    def test_epoch_end(self, outputs):
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            y_true = np.append(y_true, results_dict['y_true'])
            y_pred = np.append(y_pred, results_dict['y_pred'])
        acc = accuracy_score(y_true, y_pred)
        self.log('test_acc', acc)

## 2: Implement LightningDataModule

class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, num_workers, normalization_vector):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalization_vector = normalization_vector

    def prepare_data(self):
        MNIST(os.getcwd(), train=True, download=True)
        MNIST(os.getcwd(), train=False, download=True)

    def setup(self, stage):
        # transforms
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.normalization_vector[0],
                                 self.normalization_vector[1])
        ])

        if stage == 'fit':
            mnist_train = MNIST(os.getcwd(), train=True, transform=transform)
            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        if stage == 'test':
            self.mnist_test = MNIST(os.getcwd(), train=False, transform=transform)

    def train_dataloader(self):
        mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)
        return mnist_train

    def val_dataloader(self):
        mnist_val = DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)
        return mnist_val

    def test_dataloader(self):
        mnist_test = DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)
        return mnist_test

## 3: Implement Callbacks and Create Them

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

lr_logger = LearningRateMonitor(**LearningRateMonitor_Params)

model_checkpoint = ModelCheckpoint(**ModelCheckpoint_Params)

# Step 4: Create NeptuneLogger

from pytorch_lightning.loggers.neptune import NeptuneLogger

neptune_logger = NeptuneLogger(
    api_key="ANONYMOUS",
    project_name="shared/pytorch-lightning-integration",
    close_after_fit=False,
    experiment_name="train-on-MNIST",
    params=ALL_PARAMS,
    tags=['1.x', 'advanced'],
)

# Step 5: Pass NeptuneLogger and Callbacks to the Trainer

trainer = pl.Trainer(logger=neptune_logger,
                     checkpoint_callback=model_checkpoint,
                     callbacks=[lr_logger],
                     **Trainer_Params)

# Step 6: Run experiment

## 1: Initialize model and data objects

# init model
model = LitModel(**LightningModule_Params)

# init data
dm = MNISTDataModule(**LightningDataModule_Params)

## 2: Run training

trainer.fit(model, dm)

## 3: Run testing

trainer.test(datamodule=dm)

# Step 7: Run additional actions after train and fit

## 1: Log confusion matrix

import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix

model.freeze()
test_data = dm.test_dataloader()
y_true = np.array([])
y_pred = np.array([])

for i, (x, y) in enumerate(test_data):
    y = y.cpu().detach().numpy()
    y_hat = model.forward(x).argmax(axis=1).cpu().detach().numpy()

    y_true = np.append(y_true, y)
    y_pred = np.append(y_pred, y_hat)

fig, ax = plt.subplots(figsize=(16, 12))
plot_confusion_matrix(y_true, y_pred, ax=ax)
neptune_logger.experiment.log_image('confusion_matrix', fig)

## 2: Log model checkpoints to Neptune

for k in model_checkpoint.best_k_models.keys():
    model_name = 'checkpoints/' + k.split('/')[-1]
    neptune_logger.experiment.log_artifact(k, model_name)

## 3: Log best model checkpoint score to Neptune

neptune_logger.experiment.set_property('best_model_score', model_checkpoint.best_model_score.tolist())

## 4 Log model summary

for chunk in [x for x in str(model).split('\n')]:
    neptune_logger.experiment.log_text('model_summary', str(chunk))

## 5: Log number of GPU units used

neptune_logger.experiment.set_property('num_gpus', trainer.num_gpus)

# Step 8: Stop Neptune logger at the end

neptune_logger.experiment.stop()

# Explore Results

# tests
exp = neptune_logger.experiment.id

## check data size
### train
if len(dm.train_dataloader().dataset) != 55000:
    raise ValueError('train data size does not match')

### val
if len(dm.val_dataloader().dataset) != 5000:
    raise ValueError('val data size does not match')

### test
if len(dm.test_dataloader().dataset) != 10000:
    raise ValueError('test data size does not match')

## check logs
correct_logs = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc',
                'lr-Adam','epoch',
                'grad_2.0_norm_layer_1.weight', 'grad_2.0_norm_layer_1.bias',
                'grad_2.0_norm_layer_2.weight', 'grad_2.0_norm_layer_2.bias',
                'grad_2.0_norm_total',
                'test_misclassified_images', 'confusion_matrix', 'model_summary']

if set(neptune_logger.experiment.get_logs().keys()) != set(correct_logs):
    raise ValueError('incorrect metrics')