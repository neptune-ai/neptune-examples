# Neptune + Optuna

# Before you start

## Install dependencies

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

neptune.stop()

# Advanced Options

## Log charts and study object during sweep

## Log charts and study object after the sweep

# Explore results in the Neptune UI