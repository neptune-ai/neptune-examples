# Neptune + Keras Tuner

# Before you start

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

neptune.stop()

# Explore results in the Neptune UI