# Neptune + Optuna

# Before you start

## Install dependencies

get_ipython().system(" pip install --quiet optuna==2.3.0 neptune-client==0.4.125 neptune-contrib['monitoring']==0.24.8")

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
      'objective': 'binary',
      'metric': 'binary_logloss',
      'num_leaves': trial.suggest_int('num_leaves', 2, 256),
      'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
      'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
      'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
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

exp = neptune.get_experiment()

## check logs
correct_logs = ['run_score', 'best_so_far_run_score', 'run_parameters']

if set(exp.get_logs().keys()) != set(correct_logs):
    raise ValueError('incorrect metrics')

if int(exp.get_logs()['run_score'].x) != 99:
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