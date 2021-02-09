# Tour with Scikit-learn

# Install dependencies

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

### Stop Neptune experiment after logging scores

neptune.stop()

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

### Stop Neptune experiment after logging summary

neptune.stop()

## Automatic logging to Neptune: summary

# If you want to learn more, go to the [Neptune documentation](https://docs.neptune.ai/integrations/sklearn.html).