# Tour with Scikit-learn

# Install dependencies

get_ipython().system(' pip install --quiet scikit-learn==0.24.1 neptune-client==0.5.1 neptune-contrib[monitoring]==0.26.0')

get_ipython().system(' pip install --quiet --upgrade scikit-learn neptune-client neptune-contrib[monitoring]')

# Introduction

# Logging Scikit-learn classifier meta-data to Neptune

## Basic example

parameters = {'n_estimators': 120,
              'learning_rate': 0.12,
              'min_samples_split': 3,
              'min_samples_leaf': 2}

from sklearn.datasets import load_digits
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

gbc = GradientBoostingClassifier(**parameters)

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=28743)

gbc.fit(X_train, y_train)

### Initialize Neptune

import neptune

neptune.init('shared/sklearn-integration', api_token='ANONYMOUS')

### Create an experiment and log classifier parameters

neptune.create_experiment(params=parameters,
                          name='classification-example',
                          tags=['GradientBoostingClassifier', 'classification'])

### Log scores on test data to Neptune

from sklearn.metrics import max_error, mean_absolute_error, r2_score

y_pred = gbc.predict(X_test)

neptune.log_metric('max_error', max_error(y_test, y_pred))
neptune.log_metric('mean_absolute_error', mean_absolute_error(y_test, y_pred))
neptune.log_metric('r2_score', r2_score(y_test, y_pred))

# tests
exp = neptune.get_experiment()

### Stop Neptune experiment after logging scores

neptune.stop()

# tests
# check logs
correct_logs_set = {'max_error', 'mean_absolute_error', 'r2_score'}
from_exp_logs = set(exp.get_logs().keys())
assert correct_logs_set == from_exp_logs, '{} - incorrect logs'.format(exp)

# check parameters
assert set(exp.get_parameters().keys()) == set(parameters.keys()), '{} parameters do not match'.format(exp)

## Basic example: summary

### If you want to learn more, go to the [Neptune documentation](https://docs.neptune.ai/integrations/sklearn.html).

## Automatically log classifier summary to Neptune

### Initialize Neptune

import neptune

neptune.init('shared/sklearn-integration', api_token='ANONYMOUS')

### Create an experiment and log classifier parameters

neptune.create_experiment(params=parameters,
                          name='classification-example',
                          tags=['GradientBoostingClassifier', 'classification'])

### Log classifier summary

from neptunecontrib.monitoring.sklearn import log_classifier_summary

log_classifier_summary(gbc, X_train, X_test, y_train, y_test)

# tests
exp = neptune.get_experiment()

### Stop Neptune experiment after logging summary

neptune.stop()

# check logs
correct_logs_set = {'charts_sklearn'}
for name in ['precision', 'recall', 'fbeta_score', 'support']:
    for i in range(10):
        correct_logs_set.add('{}_class_{}_test_sklearn'.format(name, i))
from_exp_logs = set(exp.get_logs().keys())
assert correct_logs_set == from_exp_logs, '{} - incorrect logs'.format(exp)

# check sklearn parameters
assert set(exp.get_properties().keys()) == set(gbc.get_params().keys()), '{} parameters do not match'.format(exp)

# check neptune parameters
assert set(exp.get_parameters().keys()) == set(parameters.keys()), '{} parameters do not match'.format(exp)

## Automatic logging to Neptune: summary

# If you want to learn more, go to the [Neptune documentation](https://docs.neptune.ai/integrations/sklearn.html).