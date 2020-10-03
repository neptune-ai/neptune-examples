# # Neptune API tour
# 

# Introduction
# 
# This guide will show you how to:
# 
# * Install neptune-client
# * Connect Neptune to your script and create the first experiment
# * Log simple metrics to Neptune and explore the in the UI
# * Log learning curves, images and model binaries from Keras training and see those in the Neptune UI
# * Fetch the data you logged to Neptune directly into your notebook and analyze them 
# 
# By the end of it, you will run your first experiment and see it in Neptune!

# Setup
# 
# Install Neptune client
# 

pip install neptune-client

# Initialize Neptune
# 
# Connects your script to Neptune application. 
# 

import neptune

neptune.init(
    api_token="ANONYMOUS",
    project_qualified_name="shared/colab-test-run"
)

# You tell Neptune: 
# 
# * **who you are**: your Neptune API token `api_token` 
# * **where you want to send your data**: your Neptune project `project_qualified_name`.
# 
# ---
# 
# **Note:** 
# 
# 
# Instead of logging data to the public project 'shared/onboarding' as an anonymous user 'neptuner' you can log it to your own project.
# 
# To do that:
# 
# 1. Get your Neptune API token
# 
# ![image](https://neptune.ai/wp-content/uploads/get_token.gif)
# 
# 2. Pass the token to ``api_token`` argument of ``neptune.init()`` method: ``api_token=YOUR_API_TOKEN``
# 3. Pass your username to the ``project_qualified_name`` argument of the ``neptune.init()`` method: ``project_qualified_name='YOUR_USERNAME/sandbox``. Keep `/sandbox` at the end, the `sandbox` project that was automatically created for you.
# 
# For example:
# 
# ```python
# neptune.init(project_qualified_name='funky_steve/sandbox', 
#              api_token='eyJhcGlfYW908fsdf23f940jiri0bn3085gh03riv03irn',
#             )
# ```
# 
# ---

# Basic Example
# 
# Lets start with something super simple.
# 
# I will:
#  create an experiment, add a tag, and send a metric value
# 
# * create an experiment
# * log hyperparameters
# * log a metric
# * append a tag
# * stop experiment
# 
# 

neptune.create_experiment(
    name='basic-colab-example',
    params={'learning_rate':0.1}
)

neptune.log_metric('accuracy', 0.93)

neptune.append_tags(['basic', 'finished_successfully'])

# You can change the values and rerun to see your experiments appear in the dashboard.

# ---
# 
# **Note:**
#    
# When you track experiments with Neptune in Jupyter notebooks you need to explicitly stop the experiment by running `neptune.stop()`.
# 
# If you are running Neptune in regular `.py` scripts it will stop automatically when your code stops running.
# 
# ---

neptune.stop()

# All `basic-colab-example` experiments are grouped in [this dashboard view](https://ui.neptune.ai/o/shared/org/colab-test-run/experiments?viewId=8dbc02b5-c68c-4833-9b43-828678145442).
# 
# ![alt text](https://neptune.ai/wp-content/uploads/Screenshot-from-2020-03-18-11-58-14.png)
# 
# There are many other things that you can log to neptune:
# 
# * Images and charts
# * Artifacts like model weights or results
# * Text values
# * Hardware consumption
# * Code snapshots
# * and more
# 
# You can go and see all that in the [documentation](https://docs.neptune.ai/python-api/introduction.html) but you can check out the next example to see some of those.
# 
# 

# Keras classification example
# 
# Install and import your machine learning libraries

pip install keras scikit-plot

# Get the data:

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

# To log metrics after every batch and epoch let's create `NeptuneLogger` callback:

from tensorflow.keras.callbacks import Callback

class NeptuneLogger(Callback):

    def on_batch_end(self, batch, logs={}):
        for log_name, log_value in logs.items():
            neptune.log_metric(f'batch_{log_name}', log_value)

    def on_epoch_end(self, epoch, logs={}):
        for log_name, log_value in logs.items():
            neptune.log_metric(f'epoch_{log_name}', log_value)

# Now we simply need to create an experiment. 
# I will tag it with the name `advanced` and log hyperparameters `epoch_nr` and `batch_size`: 

EPOCH_NR = 6
BATCH_SIZE = 32

neptune.create_experiment(name='keras-metrics',
                          params={'epoch_nr': EPOCH_NR,
                                  'batch_size': BATCH_SIZE},
                          tags=['advanced'],
                          )

# Now we pass our `NeptuneLogger` as keras callback and thats it.

history = model.fit(x=x_train,
                    y=y_train,
                    epochs=EPOCH_NR,
                    batch_size=BATCH_SIZE,
                    validation_data=(x_test, y_test),
                    callbacks=[NeptuneLogger()])

# You can click on the experiment link above and monitor your learning curves as it is training!
# 
# ![alt text](https://neptune.ai/wp-content/uploads/monitor_training.png)
# 
# Great thing is, you can log more things if you need to during or after the training is finished.
# 
# For example, let's calculate some additional metrics on test data and log them.

import numpy as np

y_test_pred = np.asarray(model.predict(x_test))
y_test_pred_class = np.argmax(y_test_pred, axis=1)

from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_test_pred_class, average='micro')

neptune.log_metric('test_f1', f1)

# We can log diagnostic charts like confusion matrix or ROC AUC curve.

import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix, plot_roc

fig, ax = plt.subplots(figsize=(16, 12))
plot_confusion_matrix(y_test, y_test_pred_class, ax=ax)
neptune.log_image('diagnostic_charts', fig)

fig, ax = plt.subplots(figsize=(16, 12))
plot_roc(y_test, y_test_pred, ax=ax)
neptune.log_image('diagnostic_charts', fig)

# ![alt text](https://neptune.ai/wp-content/uploads/logging_charts.png)
# 
# We can also log model weights to Neptune.

model.save('my_model.h5')
neptune.log_artifact('my_model.h5')

# ![alt text](https://neptune.ai/wp-content/uploads/logging_artifacts.png)
# 
# With that you can share models with your teammates easily.

# ---
# 
# **Note:**
#    
# When you track experiments with Neptune in Jupyter notebooks you need to explicitly stop the experiment by running `neptune.stop()`.
# 
# If you are running Neptune in regular `.py` scripts it will stop automatically when your code stops running.
# 
# ---

neptune.stop()

# You can play around and run this experiment with different parameters and see results and compare them.
# 
# Like I've done [here](https://ui.neptune.ai/o/shared/org/colab-test-run/compare?shortId=%5B%22COL-11%22%2C%22COL-10%22%2C%22COL-9%22%2C%22COL-6%22%5D&viewId=f93b0ebd-6c75-4862-96f3-df1a67c08ea9&chartFilter=epoch_val_acc&legendFields=%5B%22shortId%22%2C%22epoch_val_acc%22%2C%22epoch_val_loss%22%2C%22epoch_loss%22%2C%22epoch_acc%22%5D&legendFieldTypes=%5B%22native%22%2C%22numericChannels%22%2C%22numericChannels%22%2C%22numericChannels%22%2C%22numericChannels%22%5D):
# 
# ![alt text](https://neptune.ai/wp-content/uploads/exp_comparison-1.png)
# 
# A cool thing is, once things are logged to Neptune you can access them from wherever you want.
# Let me show you. 
# 
# 

# Access data you logged programatically 
# 
# Neptune lets you fetch whatever you logged to it directly to your notebooks and scripts.
# 
# Just run:

from neptune import Session

session = Session.with_default_backend(api_token="ANONYMOUS")
my_project = session.get_project("shared/colab-test-run")

# Now that your project is *fetched* you can download the experiment dashboard data.
# 
# I will download only the experiment data with the `tag="advanced"` :

my_project.get_leaderboard(tag=['advanced']).head()

# You can also access information from the individual experiment:

exp = my_project.get_experiments(id='COL-6')[0]
exp

exp.get_numeric_channels_values("epoch_loss", "epoch_val_loss")

# You can even download artifacts from that experiment if you want to:
# 
# 

exp.download_artifact('my_model.h5','./')

ls ./

# Learn more about Neptune
# 
# Read about other Neptune features, create your free account and start logging!
# 
# [**Go to Neptune**](https://neptune.ai/?utm_source=colab&utm_medium=notebook&utm_campaign=colab-examples&utm_content=api-tour)