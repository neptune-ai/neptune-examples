# Neptune + TensorFlow / Keras

# Before we start

## Install dependencies

get_ipython().system(' pip install --quiet tensorflow==2.3.1 neptune-client==0.4.132 neptune-contrib==0.25.0')

get_ipython().system(' pip install --quiet tensorflow neptune-client neptune-contrib --upgrade')

## Import libraries

import tensorflow as tf

## Define your model, data loaders and optimizer

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
])

optimizer = tf.keras.optimizers.SGD(lr=0.005, momentum=0.4,)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

## Initialize Neptune

import neptune

neptune.init(api_token='ANONYMOUS', project_qualified_name='shared/tensorflow-keras-integration')

# Quickstart

## Step 1: Create an Experiment

neptune.create_experiment('tensorflow-keras-quickstart')

## Step 2: Add NeptuneMonitor Callback to model.fit()

from neptunecontrib.monitoring.keras import NeptuneMonitor

model.fit(x_train, y_train,
          epochs=5,
          batch_size=64,
          callbacks=[NeptuneMonitor()])

## Step 3: Explore results in the Neptune UI

## Step 4: Stop logging

# tests

exp = neptune.get_experiment()

neptune.stop()

# tests

all_logs = exp.get_logs()

## check logs
correct_logs = ['batch_loss', 'batch_accuracy', 'epoch_loss', 'epoch_accuracy']

if set(all_logs.keys()) != set(correct_logs):
    raise ValueError('incorrect metrics')

# More Options

## Log hardware consumption

get_ipython().system(' pip install --quiet psutil==5.6.6')

get_ipython().system(' pip install --quiet psutil')

## Log hyperparameters

PARAMS = {'lr':0.005, 
          'momentum':0.9, 
          'epochs':10,
          'batch_size':32}

optimizer = tf.keras.optimizers.SGD(lr=PARAMS['lr'], momentum=PARAMS['momentum'])

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# log params
neptune.create_experiment('tensorflow-keras-advanced', params=PARAMS)

model.fit(x_train, y_train,
          epochs=PARAMS['epochs'],
          batch_size=PARAMS['batch_size'],
          callbacks=[NeptuneMonitor()])

## Log image predictions

x_test_sample = x_test[:100]
y_test_sample_pred = model.predict(x_test_sample)

for image, y_pred in zip(x_test_sample, y_test_sample_pred):
    description = '\n'.join(['class {}: {}'.format(i, pred)
                                for i, pred in enumerate(y_pred)])
    neptune.log_image('predictions',
                      image,
                      description=description)

## Log model weights

model.save('my_model')

# log model
neptune.log_artifact('my_model')

# Explore results in the Neptune UI

## Stop logging

# tests

exp = neptune.get_experiment()

neptune.stop()

# tests

all_logs = exp.get_logs()

## check logs
correct_logs = ['batch_loss', 'batch_accuracy', 'epoch_loss', 'epoch_accuracy', 'predictions']

if set(all_logs.keys()) != set(correct_logs):
    raise ValueError('incorrect metrics')