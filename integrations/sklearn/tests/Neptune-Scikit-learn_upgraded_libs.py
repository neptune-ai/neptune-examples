# Scikit-learn + Neptune

# Before you start

## Install dependencies

get_ipython().system(' pip install scikit-learn==0.23.2 neptune-client==0.4.129 neptune-contrib==0.25.0')

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

neptune.init('shared/sklearn-integration',
             api_token='ANONYMOUS')
exp = neptune.create_experiment(name='regression-example',
                                tags=['RandomForestRegressor', 'regression'])

log_regressor_summary(rfr, X_train, X_test, y_train, y_test, experiment=exp)

# tests
# check logs
correct_logs_set = {'evs_test_sklearn', 'me_test_sklearn', 'mae_test_sklearn', 'r2_test_sklearn', 'charts_sklearn'}
from_exp_logs = set(exp.get_logs().keys())
assert correct_logs_set == from_exp_logs, '{} - incorrect logs'.format(exp)

# check parameters
assert set(exp.get_properties().keys()) == set(rfr.get_params().keys()), '{} parameters do not match'.format(exp)

# check estimator
import joblib
import os
import tempfile

import numpy as np

with tempfile.TemporaryDirectory() as d:
    exp.download_artifact('model/estimator.skl', d)
    full_path = os.path.join(d, 'estimator.skl')
    model = joblib.load(full_path)

assert np.array_equal(model.predict(X_test), rfr.predict(X_test)), '{} estimator error'.format(exp)

## Step 4: Stop Neptune experiment after logging summary

exp.stop()

## Explore results

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

neptune.init('shared/sklearn-integration',
             api_token='ANONYMOUS')
exp = neptune.create_experiment(name='classification-example',
                                tags=['GradientBoostingClassifier', 'classification'])

log_classifier_summary(gbc, X_train, X_test, y_train, y_test, experiment=exp)

# tests
# check logs
correct_logs_set = {'charts_sklearn'}
for name in ['precision', 'recall', 'fbeta_score', 'support']:
    for i in range(10):
        correct_logs_set.add('{}_class_{}_test_sklearn'.format(name, i))
from_exp_logs = set(exp.get_logs().keys())
assert correct_logs_set == from_exp_logs, '{} - incorrect logs'.format(exp)

# check parameters
assert set(exp.get_properties().keys()) == set(gbc.get_params().keys()), '{} parameters do not match'.format(exp)

# check estimator
import joblib
import os
import tempfile

import numpy as np

with tempfile.TemporaryDirectory() as d:
    exp.download_artifact('model/estimator.skl', d)
    full_path = os.path.join(d, 'estimator.skl')
    model = joblib.load(full_path)

assert np.array_equal(model.predict_proba(X_test), gbc.predict_proba(X_test)), '{} estimator error'.format(exp)

## Step 4: Stop Neptune experiment after logging summary

exp.stop()

## Explore Results

# Scikit-learn KMeans clustering

## Step 1: Import libraries

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

## Step 2: Create KMeans object and example data

km = KMeans(n_init=11, max_iter=270)

X, y = make_blobs(n_samples=579, n_features=17, centers=7, random_state=28743)

## Step 3: Create Neptune experiment and log KMeans clustering summary

import neptune
from neptunecontrib.monitoring.sklearn import log_kmeans_clustering_summary

neptune.init('shared/sklearn-integration',
             api_token='ANONYMOUS')
exp = neptune.create_experiment(name='clustering-example',
                                tags=['KMeans', 'clustering'])

log_kmeans_clustering_summary(km, X, n_clusters=17, experiment=exp)

# tests
# check logs
assert list(exp.get_logs().keys()) == ['charts_sklearn'], '{} - incorrect logs'.format(exp)

# check cluster labels
assert X.shape[0] == len(km.labels_), '{} incorrect number of cluster labels'.format(exp)

# check parameters
assert set(exp.get_properties().keys()) == set(km.get_params().keys()), '{} parameters do not match'.format(exp)

## Step 4: Stop Neptune experiment after logging summary

exp.stop()

## Explore Results

# tests
# run example regression, classification and clustering jobs
import neptune
from sklearn.cluster import KMeans
from sklearn.datasets import load_boston, load_digits, load_linnerud, make_blobs
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression, Perceptron, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier

from neptunecontrib.monitoring.sklearn import log_regressor_summary, log_classifier_summary,     log_kmeans_clustering_summary

neptune.init('shared/sklearn-integration',
             api_token='ANONYMOUS')

def run_regressors_single_output():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=28743)

    gpr = GaussianProcessRegressor()
    train_regressor(gpr, 'GaussianProcessRegressor', 'single-output', X_train, X_test, y_train, y_test)

    gbr = GradientBoostingRegressor()
    train_regressor(gbr, 'GradientBoostingRegressor', 'single-output', X_train, X_test, y_train, y_test)

    lr = LinearRegression()
    train_regressor(lr, 'LinearRegression', 'single-output', X_train, X_test, y_train, y_test)

def run_regressors_multi_output():
    X, y = load_linnerud(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    mor = MultiOutputRegressor(Ridge())
    train_regressor(mor, 'MultiOutputRegressor', 'multi-output', X_train, X_test, y_train, y_test)

def train_regressor(reg, name, tag, X_train, X_test, y_train, y_test):
    neptune.create_experiment(tags=[name, tag, 'regression'])
    reg.fit(X_train, y_train)
    log_regressor_summary(reg, X_train, X_test, y_train, y_test)
    neptune.stop()

def run_classifiers():
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=28743)

    rfc = RandomForestClassifier()
    train_classifier(rfc, 'RandomForestClassifier', 'single-output', X_train, X_test, y_train, y_test)

    knc = KNeighborsClassifier()
    train_classifier(knc, 'KNeighborsClassifier', 'single-output', X_train, X_test, y_train, y_test)

    lr = LogisticRegression()
    train_classifier(lr, 'LogisticRegression', 'single-output', X_train, X_test, y_train, y_test)

    p = Perceptron()
    train_classifier(p, 'Perceptron', 'single-output', X_train, X_test, y_train, y_test)

    rc = RidgeClassifier()
    train_classifier(rc, 'RidgeClassifier', 'single-output', X_train, X_test, y_train, y_test)

def train_classifier(clf, name, tag, X_train, X_test, y_train, y_test):
    neptune.create_experiment(tags=[name, tag, 'classifier'])
    clf.fit(X_train, y_train)
    log_classifier_summary(clf, X_train, X_test, y_train, y_test)
    neptune.stop()

def run_clustering():
    X, y = make_blobs(n_samples=579, n_features=17, centers=7, random_state=28743)
    km = KMeans()
    train_clustering(km, 'KMeans', 'clustering', X)

def train_clustering(model, name, tag, data):
    neptune.create_experiment(tags=[name, tag, 'clustering'])
    log_kmeans_clustering_summary(model, X)
    neptune.stop()

run_regressors_single_output()
run_regressors_multi_output()
run_classifiers()
run_clustering()