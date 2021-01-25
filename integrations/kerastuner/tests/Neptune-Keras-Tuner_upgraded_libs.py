# Neptune + Keras Tuner

# Before you start

## Install dependencies

get_ipython().system(' pip install --quiet keras-tuner==1.0.2 tensorflow==2.4.1 plotly==4.14.3 neptune-client==0.4.133 neptune-contrib[monitoring]==0.26.0')

get_ipython().system(' pip install --quiet keras-tuner tensorflow plotly neptune-client neptune-contrib[monitoring] --upgrade')

## Import libraries and prepare dataset

from tensorflow import keras
from tensorflow.keras import layers

from kerastuner.tuners import  BayesianOptimization

(x, y), (val_x, val_y) = keras.datasets.mnist.load_data()
x = x.astype('float32') / 255.
val_x = val_x.astype('float32') / 255.

x = x[:10000]
y = y[:10000]

## Create a model building function

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    for i in range(hp.Int('num_layers', 2, 20)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i), 32, 512, 32),
                               activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model

## Initialize Neptune

import neptune

neptune.init(api_token='ANONYMOUS', project_qualified_name='shared/keras-tuner-integration')

# Quickstart

## Step 1: Create an Experiment

neptune.create_experiment('bayesian-sweep')

## Step 2: Pass Neptune Logger to Tuner

import neptunecontrib.monitoring.kerastuner as npt_utils 

tuner =  BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    num_initial_points=3,
    executions_per_trial=3,
    project_name='bayesian-sweep',
    logger=npt_utils.NeptuneLogger())

## Step 3: Run the search and monitor it in Neptune 

tuner.search(x=x,
             y=y,
             epochs=5,
             validation_data=(val_x, val_y))

## Step 4: Log additional sweep information after the sweep

npt_utils.log_tuner_info(tuner)

## Step 5: Stop logging

# tests
exp = neptune.get_experiment()

neptune.stop()

# tests
all_logs = exp.get_logs()

## check logs
correct_logs = ['val_accuracy', 'hyperparameters/values', 'val_loss', 'loss', 'accuracy', 'run_score', 'hyperparameters/space', 'best_score']

assert set(all_logs.keys()) == set(correct_logs), 'Expected: {}. Actual: {}'.format(set(correct_logs), set(all_logs.keys()))

all_properties = exp.get_properties()

## check logs
correct_properties = ['objective/name', 'best_trial_id', 'best_parameters', 'tuner_id', 'objective/direction']

assert set(all_properties.keys()) == set(correct_properties), 'Expected: {}. Actual: {}'.format(set(correct_properties), set(all_properties.keys()))

# Explore results in the Neptune UI