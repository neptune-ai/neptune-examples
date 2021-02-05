# Neptune + LightGBM

# Before you start

## Install Dependencies

get_ipython().system(' pip install --quiet lightgbm==2.2.3 neptune-client==0.4.132 neptune-contrib[monitoring]==0.25.0')

get_ipython().system(' pip install --quiet lightgbm neptune-client neptune-contrib[monitoring] --upgrade')

## Import Libraries

import lightgbm as lgb
import neptune
from neptunecontrib.monitoring.lightgbm import neptune_monitor

## Initialize Neptune. 

neptune.init(api_token='ANONYMOUS', project_qualified_name='shared/LightGBM-integration')

# Quickstart

## Step 1: Create an Experiment

neptune.create_experiment(name='LightGBM-training')

## Step 2: Pass ``neptune_monitor`` to ``lgb.train``ll

# Setting up a samplt lightGBM training job
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.1)
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {'boosting_type': 'gbdt',
              'objective': 'multiclass',
              'num_class': 3,
              'num_leaves': 31,
              'learning_rate': 0.05,
              'feature_fraction': 0.9
              }

# Passing `neptune_monitor` to `lgb.train()`
gbm = lgb.train(params,
    lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_train, lgb_eval],
    valid_names=['train','valid'],
    callbacks=[neptune_monitor()],
    )

## Step 3: Stop logging.

# tests

exp = neptune.get_experiment()

neptune.stop()

# tests

all_logs = exp.get_logs()

## check logs
correct_logs = ['train_multi_logloss', 'train_multi_logloss', 'valid_multi_logloss', 'valid_multi_logloss']

if set(all_logs.keys()) != set(correct_logs):
    raise ValueError('incorrect metrics')

# More Options

## Log hardware consumption

get_ipython().system(' pip install --quiet psutil==5.6.6')

get_ipython().system(' pip install --quiet psutil')

## Log Hyperparameters

params = {'boosting_type': 'gbdt',
              'objective': 'multiclass',
              'num_class': 3,
              'num_leaves': 31,
              'learning_rate': 0.05,
              'feature_fraction': 0.9
              }

# Log hyperparameters
neptune.create_experiment(name='LightGBM-training', params=params)

# Train a model

gbm = lgb.train(params,
    lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_train, lgb_eval],
    valid_names=['train','valid'],
    callbacks=[neptune_monitor()],
    )

## Save Model Artifacts.

gbm.save_model('lightgbm.pkl')

# Log model
neptune.log_artifact('lightgbm.pkl')

## Log Interactive Charts.

### 1. Install dependencies

get_ipython().system(' pip install --quiet scikit-plot matplotlib==3.2.0 plotly==4.12.0')

### 2. Create an ROC AUC curve

import matplotlib.pyplot as plt
from scikitplot.metrics import plot_roc

y_test_pred = gbm.predict(X_test)

fig, ax = plt.subplots()
plot_roc(y_test, y_test_pred, ax=ax)

### 3. Log it to Neptune via `log_chart()` function.

from neptunecontrib.api import log_chart

log_chart(name='ROC curve', chart=fig)

## Stop logging

# tests

exp = neptune.get_experiment()

neptune.stop()

# tests

all_logs = exp.get_logs()

## check logs
correct_logs = ['train_multi_logloss', 'train_multi_logloss', 'valid_multi_logloss', 'valid_multi_logloss']

if set(all_logs.keys()) != set(correct_logs):
    raise ValueError('incorrect metrics')

# Explore results in the Neptune UI