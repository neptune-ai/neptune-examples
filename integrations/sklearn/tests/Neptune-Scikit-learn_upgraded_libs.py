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
neptune.create_experiment(name='regression-example',
                          tags=['RandomForestRegressor', 'regression'])

log_regressor_summary(rfr, X_train, X_test, y_train, y_test)

# tests
exp = neptune.get_experiment()

# check logs
correct_logs_set = {'evs_sklearn', 'me_sklearn', 'mae_sklearn', 'r2_sklearn', 'charts_sklearn'}
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

neptune.stop() # close experiment after logging summary

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

neptune.init('shared/sklearn-integration')
neptune.create_experiment(name='classification-example',
                          tags=['GradientBoostingClassifier', 'classification'])

log_classifier_summary(gbc, X_train, X_test, y_train, y_test)

# tests
exp = neptune.get_experiment()

# check logs
correct_logs_set = {'charts_sklearn'}
for name in ['precision', 'recall', 'fbeta_score', 'support']:
    for i in range(10):
        correct_logs_set.add('{}_class_{}_sklearn'.format(name, i))
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

neptune.stop() # close experiment after logging summary

## Explore Results

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
neptune.create_experiment(name='clustering-example',
                          tags=['KMeans', 'clustering'])

log_kmeans_clustering_summary(km, data=X, k=11)

# tests
exp = neptune.get_experiment()

# check logs
assert list(exp.get_logs().keys()) == ['charts_sklearn'], '{} - incorrect logs'.format(exp)

# check cluster labels
assert X.shape[0] == len(km.labels_), '{} incorrect number of cluster labels'.format(exp)

# check parameters
assert set(exp.get_properties().keys()) == set(km.get_params().keys()), '{} parameters do not match'.format(exp)

neptune.stop() # close experiment after logging summary

## Explore Results

# tests
# run example regression, classification and clustering jobs
import neptune
from sklearn.cluster import KMeans
from sklearn.datasets import load_boston, load_digits, load_linnerud, make_blobs
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor,     GradientBoostingRegressor, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier,     BaggingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge, LinearRegression, MultiTaskElasticNet, MultiTaskLasso,     LogisticRegression, Perceptron, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from neptunecontrib.monitoring.sklearn import log_regressor_summary, log_classifier_summary,     log_kmeans_clustering_summary

neptune.init('shared/sklearn-integration')

def run_regressors_single_output():
    # Data
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=28743)

    # Regressors, training, logging
    tag = 'single-output'

    abr = AdaBoostRegressor()
    train_regressor(abr, 'AdaBoostRegressor', tag, X_train, X_test, y_train, y_test)

    etr = ExtraTreesRegressor()
    train_regressor(etr, 'ExtraTreesRegressor', tag, X_train, X_test, y_train, y_test)

    gpr = GaussianProcessRegressor()
    train_regressor(gpr, 'GaussianProcessRegressor', tag, X_train, X_test, y_train, y_test)

    gbr = GradientBoostingRegressor()
    train_regressor(gbr, 'GradientBoostingRegressor', tag, X_train, X_test, y_train, y_test)

    lr = LinearRegression()
    train_regressor(lr, 'LinearRegression', tag, X_train, X_test, y_train, y_test)

    rfr = RandomForestRegressor()
    train_regressor(rfr, 'RandomForestRegressor', tag, X_train, X_test, y_train, y_test)

    r = Ridge()
    train_regressor(r, 'Ridge', tag, X_train, X_test, y_train, y_test)

def run_regressors_multi_output():
    # Data
    X, y = load_linnerud(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # Regressors, training, logging
    tag = 'multi-output'

    mor = MultiOutputRegressor(Ridge())
    train_regressor(mor, 'MultiOutputRegressor', tag, X_train, X_test, y_train, y_test)

    mten = MultiTaskElasticNet()
    train_regressor(mten, 'MultiTaskElasticNet', tag, X_train, X_test, y_train, y_test)

    mtl = MultiTaskLasso()
    train_regressor(mtl, 'MultiTaskLasso', tag, X_train, X_test, y_train, y_test)

def train_regressor(reg, name, tag, X_train, X_test, y_train, y_test):
    neptune.create_experiment(tags=[name, tag, 'regression'])
    reg.fit(X_train, y_train)
    log_regressor_summary(reg, X_train, X_test, y_train, y_test)
    neptune.stop()

def run_classifiers():
    # Data
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=28743)

    # Classifier, training, logging
    tag = 'single-output'

    abc = AdaBoostClassifier()
    train_classifier(abc, 'AdaBoostClassifier', tag, X_train, X_test, y_train, y_test)

    rfc = RandomForestClassifier()
    train_classifier(rfc, 'RandomForestClassifier', tag, X_train, X_test, y_train, y_test)

    bc = BaggingClassifier()
    train_classifier(bc, 'BaggingClassifier', tag, X_train, X_test, y_train, y_test)

    dtc = DecisionTreeClassifier()
    train_classifier(dtc, 'DecisionTreeClassifier', tag, X_train, X_test, y_train, y_test)

    etc = ExtraTreeClassifier()
    train_classifier(etc, 'ExtraTreeClassifier', tag, X_train, X_test, y_train, y_test)

    etsc = ExtraTreesClassifier()
    train_classifier(etsc, 'ExtraTreesClassifier', tag, X_train, X_test, y_train, y_test)

    gnb = GaussianNB()
    train_classifier(gnb, 'GaussianNB', tag, X_train, X_test, y_train, y_test)

    gbc = GradientBoostingClassifier()
    train_classifier(gbc, 'GradientBoostingClassifier', tag, X_train, X_test, y_train, y_test)

    knc = KNeighborsClassifier()
    train_classifier(knc, 'KNeighborsClassifier', tag, X_train, X_test, y_train, y_test)

    lr = LogisticRegression()
    train_classifier(lr, 'LogisticRegression', tag, X_train, X_test, y_train, y_test)

    p = Perceptron()
    train_classifier(p, 'Perceptron', tag, X_train, X_test, y_train, y_test)

    rc = RidgeClassifier()
    train_classifier(rc, 'RidgeClassifier', tag, X_train, X_test, y_train, y_test)

def train_classifier(clf, name, tag, X_train, X_test, y_train, y_test):
    neptune.create_experiment(tags=[name, tag, 'classifier'])
    clf.fit(X_train, y_train)
    log_classifier_summary(clf, X_train, X_test, y_train, y_test)
    neptune.stop()

def run_clustering():
    # Data
    X, y = make_blobs(n_samples=579, n_features=17, centers=7, random_state=28743)

    # Clustering, training, logging
    tag = 'clustering'

    km = KMeans()
    train_clustering(km, 'KMeans', tag, X)

def train_clustering(model, name, tag, data):
    neptune.create_experiment(tags=[name, tag, 'clustering'])
    log_kmeans_clustering_summary(model, data=data)
    neptune.stop()

run_regressors_single_output()
run_regressors_multi_output()
run_classifiers()
run_clustering()