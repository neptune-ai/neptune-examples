# Neptune + TensorBoard

# Convert TensorBoard logs to Neptune experiments

# Before you start

## Install Dependencies

get_ipython().system(' pip install --quiet tensorboard==2.4.0 tensorflow==2.3.1 neptune-tensorboard==0.5.1 neptune-client==0.4.132')

get_ipython().system(' pip install --quiet --upgrade tensorboard neptune-tensorboard neptune-client')

## Create some TensorBoard logs

import tensorflow as tf
import datetime

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
      ])

model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=x_train, 
          y=y_train, 
          epochs=5, 
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard_callback])

# Step 1: Set your Environment Variables

get_ipython().run_line_magic('env', 'NEPTUNE_API_TOKEN=ANONYMOUS')
get_ipython().run_line_magic('env', 'NEPTUNE_PROJECT=shared/tensorboard-integration')

# Step 2: Convert TensorBoard logs to Neptune experiments

get_ipython().system(' neptune tensorboard logs')

# See converted experiments
# 
# Click on the link(s) above to browse the TensorBoard run in Neptune or go to [shared/tensorflow-integration project](https://ui.neptune.ai/o/shared/org/tensorboard-integration/experiments?viewId=def2c858-3510-4bf9-9e52-8720fadecb11).

# Log runs live to Neptune via TensorBoard

# Step 1: Initialize Neptune

import neptune

neptune.init(api_token='ANONYMOUS', project_qualified_name='shared/tensorboard-integration')

# Step 2: Create an experiment

neptune.create_experiment('tensorboard-logging')

# Step 3: Run ``neptune_tensorboard.integrate_with_tensorflow()``

import neptune_tensorboard

neptune_tensorboard.integrate_with_tensorflow()

# Step 4: Add your training code

import tensorflow as tf
import datetime

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
      ])

model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=x_train, 
          y=y_train, 
          epochs=5, 
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard_callback])

# Step 5: Explore results in the Neptune UI

# tests

exp = neptune.get_experiment()

neptune.stop()

# tests

all_logs = exp.get_logs()

## check logs
correct_logs = ['loss', 'accuracy', 'val_loss', 'val_accuracy']

if set(all_logs.keys()) != set(correct_logs):
    raise ValueError('incorrect metrics')

# More logging options

# Create an experiment and train a model

neptune.create_experiment('tensorboard-more-logging-options')
neptune_tensorboard.integrate_with_tensorflow()

model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=x_train, 
          y=y_train, 
          epochs=5, 
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard_callback])

# Log model weights

model.save('my_model')

neptune.log_artifact('my_model')

# Log interactive charts

## Install neptune-contrib

get_ipython().system(' pip install --quiet neptune-contrib==0.24.9 matplotlib==3.2.0 scikit-plot==0.3.7 plotly==4.12.0')

get_ipython().system(' pip install --quiet neptune-contrib scikit-plot --upgrade')

## Create chart

import matplotlib.pyplot as plt 
from scikitplot.metrics import plot_roc

y_test_pred = model.predict(x_test)

fig, ax = plt.subplots()
plot_roc(y_test, y_test_pred, ax=ax)

## Log chart to Neptune as interactive Plotly chart

from neptunecontrib.api import log_chart

log_chart(name='ROC curve', chart=fig)

# tests

exp = neptune.get_experiment()

neptune.stop()

# tests

all_logs = exp.get_logs()

## check logs
correct_logs = ['loss', 'accuracy', 'val_loss', 'val_accuracy']

if set(all_logs.keys()) != set(correct_logs):
    raise ValueError('incorrect metrics')