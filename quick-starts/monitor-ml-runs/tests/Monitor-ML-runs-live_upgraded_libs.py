# Monitor ML runs live 

# Introduction
# 
# This guide will show you how to:
# 
# * Monitor training and evaluation metrics and losses live
# * Monitor hardware resources during training
# 
# By the end of it, you will monitor your metrics, losses, and hardware live in Neptune!

# Setup

get_ipython().system(' pip install neptune-client==0.4.130 tensorflow==2.3.0')

get_ipython().system(' pip install neptune-client tensorflow --upgrade')

# Step 1: Create a basic training script

from tensorflow import keras

# parameters
PARAMS = {'epoch_nr': 10,
          'batch_size': 256,
          'lr': 0.005,
          'momentum': 0.4,
          'use_nesterov': True,
          'unit_nr': 256,
          'dropout': 0.05}

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(PARAMS['unit_nr'], activation=keras.activations.relu),
    keras.layers.Dropout(PARAMS['dropout']),
    keras.layers.Dense(10, activation=keras.activations.softmax)
])

optimizer = keras.optimizers.SGD(lr=PARAMS['lr'],
                                 momentum=PARAMS['momentum'],
                                 nesterov=PARAMS['use_nesterov'], )

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 2: Initialize Neptune

import neptune

neptune.init(project_qualified_name='shared/onboarding', # change this to your `workspace_name/project_name`
             api_token='ANONYMOUS', # change this to your api token
            )

# Step 3: Create an experiment

neptune.create_experiment(name='great-idea')

# Step 4: Add logging for metrics and losses

class NeptuneMonitor(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for metric_name, metric_value in logs.items():
            neptune.log_metric(metric_name, metric_value)

model.fit(x_train, y_train,
          epochs=PARAMS['epoch_nr'],
          batch_size=PARAMS['batch_size'],
          callbacks=[NeptuneMonitor()])

# tests
current_exp = neptune.get_experiment()

correct_logs = ['loss', 'accuracy']

if set(current_exp.get_logs().keys()) != set(correct_logs):
    raise ValueError()