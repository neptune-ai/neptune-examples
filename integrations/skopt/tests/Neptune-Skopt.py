# Neptune + Scikit-Optimize

# Before you start

## Install dependencies

get_ipython().system(" pip install --quiet scikit-optimize==0.8.1 neptune-client==0.4.130 neptune-contrib['monitoring']==0.25.0")

## Create a sample objective function for skopt

import skopt
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

space = [skopt.space.Real(0.01, 0.5, name='learning_rate', prior='log-uniform'),
          skopt.space.Integer(1, 30, name='max_depth'),
          skopt.space.Integer(2, 100, name='num_leaves'),
          skopt.space.Integer(10, 1000, name='min_data_in_leaf'),
          skopt.space.Real(0.1, 1.0, name='feature_fraction', prior='uniform'),
          skopt.space.Real(0.1, 1.0, name='subsample', prior='uniform'),
          ]

@skopt.utils.use_named_args(space)
def objective(**params):
    data, target = load_breast_cancer(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.25)
    dtrain = lgb.Dataset(train_x, label=train_y)

    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity':-1,
        **params
    }

    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(test_x)
    accuracy = roc_auc_score(test_y, preds)
    return -1.0 * accuracy

# Quickstart

## Step 1: Initialize Neptune

import neptune

neptune.init(api_token='ANONYMOUS', project_qualified_name='shared/scikit-optimize-integration')

## Step 2: Create an Experiment

neptune.create_experiment(name='skopt-sweep')

## Step 3: Run skopt with the Neptune Callback

# Create Neptune Callback
import neptunecontrib.monitoring.skopt as skopt_utils

neptune_callback = skopt_utils.NeptuneCallback()

# Run the skopt minimize function with the Neptune Callback
results = skopt.forest_minimize(objective, space, n_calls=25, n_random_starts=10,
                                callback=[neptune_callback])

## Step 4: Log best parameter configuration, best score and diagnostic plots

skopt_utils.log_results(results)

## Step 5: Stop logging and Explore results in the Neptune UI

# tests

exp = neptune.get_experiment()

neptune.stop()

# tests

all_logs = exp.get_logs()

## check logs
correct_logs = ['run_score', 'best_so_far_run_score', 'best_score', 'run_parameters', 'diagnostics']

assert set(all_logs.keys()) == set(correct_logs), 'Expected: {}. Actual: {}'.format(set(correct_logs), set(all_logs.keys()))