# Neptune + Optuna

# Before you start

## Install dependencies

get_ipython().system(' pip install --quiet optuna==2.3.0 lightgbm==3.1.0 plotly==4.13.0 neptune-client==0.4.126 neptune-contrib[monitoring]==0.24.9')

## Import libraries

import lightgbm as lgb
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

## Create a sample `objective` function for Optuna

def objective(trial):
   
   data, target = load_breast_cancer(return_X_y=True)
   train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.25)
   dtrain = lgb.Dataset(train_x, label=train_y)

   param = {
      'verbose': -1,
      'objective': 'binary',
      'metric': 'binary_logloss',
      'num_leaves': trial.suggest_int('num_leaves', 2, 256),
      'feature_fraction': trial.suggest_uniform('feature_fraction', 0.2, 1.0),
      'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.2, 1.0),
      'min_child_samples': trial.suggest_int('min_child_samples', 3, 100),
   }

   gbm = lgb.train(param, dtrain)
   preds = gbm.predict(test_x)
   accuracy = roc_auc_score(test_y, preds)
   
   return accuracy

## Initialize Neptune

import neptune

neptune.init(api_token='ANONYMOUS', project_qualified_name='shared/optuna-integration')

# Quickstart

## Step 1: Create an Experiment

neptune.create_experiment('optuna-sweep')

## Step 2: Create the Neptune Callback

import neptunecontrib.monitoring.optuna as opt_utils

neptune_callback = opt_utils.NeptuneCallback()

## Step 3: Run Optuna with the Neptune Callback

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, callbacks=[neptune_callback])

## Step 4: Stop logging

# tests

import time

time.sleep(5)

exp = neptune.get_experiment()
all_logs = exp.get_logs()

## check logs
correct_logs = ['run_score', 'best_so_far_run_score', 'run_parameters']

if set(all_logs.keys()) != set(correct_logs):
    raise ValueError('incorrect metrics')

## check run_parameters
correct_parameters = ['num_leaves', 'feature_fraction', 'bagging_fraction', 'min_child_samples']

run_parameters = eval(all_logs['run_parameters'].y)

if run_parameters.keys() != set(correct_parameters):
    raise ValueError('incorrect run parameters')

if int(all_logs['run_score'].x) != 99:
    print(int(all_logs['run_score'].x))
    raise ValueError('wrong number of iterations logged')

neptune.stop()

# Advanced Options

## Log charts and study object during sweep

# Create experiment
neptune.create_experiment('optuna-sweep-advanced')

# Create callback to log advanced options during the sweep
neptune_callback = opt_utils.NeptuneCallback(log_study=True, log_charts=True)

# Run Optuna with Neptune Callback
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20, callbacks=[neptune_callback])

# Stop logging 
neptune.stop()

## Log charts and study object after the sweep

# Create experiment
neptune.create_experiment('optuna-sweep-advanced')

# Create Neptune callback
neptune_callback = opt_utils.NeptuneCallback()

# Run Optuna with Neptune Callback
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, callbacks=[neptune_callback])

# Log Optuna charts and study object after the sweep is complete
opt_utils.log_study_info(study)

# Stop logging 
neptune.stop()

# Explore results in the Neptune UI