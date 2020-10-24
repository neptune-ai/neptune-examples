# Neptune API tour
# 

# Setup

get_ipython().system(' pip install neptune-client==0.4.123')

# Initialize Neptune

import neptune

neptune.init(
    api_token="ANONYMOUS",
    project_qualified_name="shared/colab-test-run"
)

# Basic Example

neptune.create_experiment(
    name='basic-colab-example',
    params={'learning_rate':0.1}
)

neptune.log_metric('accuracy', 0.93)

neptune.append_tags(['basic', 'finished_successfully'])

# tests
current_exp = neptune.get_experiment()

if set(current_exp.get_logs().keys()) != set(['accuracy']):
    raise ValueError()

neptune.stop()

# Keras classification example [Advanced]

get_ipython().system('pip install tensorflow==2.3.0 scikit-plot==0.3.7 --user')

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

from tensorflow.keras.callbacks import Callback

class NeptuneLogger(Callback):

    def on_batch_end(self, batch, logs={}):
        for log_name, log_value in logs.items():
            neptune.log_metric(f'batch_{log_name}', log_value)

    def on_epoch_end(self, epoch, logs={}):
        for log_name, log_value in logs.items():
            neptune.log_metric(f'epoch_{log_name}', log_value)

EPOCH_NR = 6
BATCH_SIZE = 32

neptune.create_experiment(name='keras-metrics',
                          params={'epoch_nr': EPOCH_NR,
                                  'batch_size': BATCH_SIZE},
                          tags=['advanced'],
                          )

history = model.fit(x=x_train,
                    y=y_train,
                    epochs=EPOCH_NR,
                    batch_size=BATCH_SIZE,
                    validation_data=(x_test, y_test),
                    callbacks=[NeptuneLogger()])

import numpy as np

y_test_pred = np.asarray(model.predict(x_test))
y_test_pred_class = np.argmax(y_test_pred, axis=1)

from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_test_pred_class, average='micro')

neptune.log_metric('test_f1', f1)

import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix, plot_roc

fig, ax = plt.subplots(figsize=(16, 12))
plot_confusion_matrix(y_test, y_test_pred_class, ax=ax)
neptune.log_image('diagnostic_charts', fig)

fig, ax = plt.subplots(figsize=(16, 12))
plot_roc(y_test, y_test_pred, ax=ax)
neptune.log_image('diagnostic_charts', fig)

model.save('my_model.h5')
neptune.log_artifact('my_model.h5')

# tests
current_exp = neptune.get_experiment()

correct_logs = ['batch_loss', 'batch_accuracy', 'epoch_loss', 
                'epoch_accuracy', 'epoch_val_loss', 'epoch_val_accuracy', 
                'test_f1', 'diagnostic_charts']

if set(current_exp.get_logs().keys()) != set(correct_logs):
    raise ValueError()

neptune.stop()

# Access data you logged programatically 

my_project = neptune.init(api_token="ANONYMOUS", project_qualified_name="shared/colab-test-run")

my_project.get_leaderboard(tag=['advanced']).head()

exp = my_project.get_experiments(id='COL-6')[0]
exp

exp.get_numeric_channels_values("epoch_loss", "epoch_val_loss")