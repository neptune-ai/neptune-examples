# Scikit-learn + Neptune

# Before you start

## Install dependencies

get_ipython().system(' pip install scikit-learn==0.23.2 neptune-client==0.4.126 neptune-contrib==0.24.9')

get_ipython().system(' pip install scikit-learn neptune-client neptune-contrib --upgrade')

# Scikit-learn regression

## Step 1: Import libraries

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

## Step 2: Create and fit random forest regressor

rfr = RandomForestRegressor(n_estimators=70, max_depth=7, min_samples_split=3)

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=28743)

rfr.fit(X_train, y_train)

## Step 3: Create Neptune experiment and log regressor summary

import neptune
from neptunecontrib.monitoring.sklearn import log_regressor_summary

neptune.init('shared/sklearn-integration')
neptune.create_experiment(tags=['RandomForestRegressor', 'regression'])

log_regressor_summary(rfr, X_train, X_test, y_train, y_test)

neptune.stop() # close experiment after logging summary

## Explore results

# tests
# check properties names if correct
# download pickled model and load it, and socre it on test
# check logs names

# Scikit-learn classification

## Step 1: Import libraries

from sklearn.datasets import load_digits
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

## Step 2: Create and fit gradient boosting classifier

gbc = GradientBoostingClassifier(n_estimators=120, learning_rate=0.12, min_samples_split=3, min_samples_leaf=2)

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=28743)

gbc.fit(X_train, y_train)

## Step 3: Create Neptune experiment and log regressor summary

import neptune
from neptunecontrib.monitoring.sklearn import log_classifier_summary

neptune.init('shared/sklearn-integration')
neptune.create_experiment(tags=['GradientBoostingClassifier', 'classification'])

log_classifier_summary(gbc, X_train, X_test, y_train, y_test)

neptune.stop() # close experiment after logging summary

## Explore Results

# tests
# check properties names if correct
# download pickled model and load it, and socre it on test
# check logs names

# Scikit-learn kmeans clustering

## Step 1: Import libraries

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

## Step 2: Create KMeans object and example data

km = KMeans(n_init=11, max_iter=270)

X, y = make_blobs(n_samples=579, n_features=17, centers=7, random_state=28743)

## Step 3: Create Neptune experiment and log KMeans clustering summary

import neptune
from neptunecontrib.monitoring.sklearn import log_kmeans_clustering_summary

neptune.init('shared/sklearn-integration')
neptune.create_experiment(tags=['KMeans', 'clustering'])

log_kmeans_clustering_summary(km, k=11, data=X)

neptune.stop() # close experiment after logging summary

## Explore Results

# tests
# check properties names if correct
# download cluster labels, check size
# check logs names