# XGBoost + Neptune integration

# Before you start

## Install dependencies

get_ipython().system(' pip install --user neptune-client==0.4.124 neptune-contrib[monitoring]==0.24.3 xgboost==1.2.0 pandas==1.0.5 scikit-learn==0.23.2')

get_ipython().system(' pip install --user --upgrade neptune-client neptune-contrib[monitoring] xgboost pandas scikit-learn')

import neptune
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from neptunecontrib.monitoring.xgboost import neptune_callback

# Set project

neptune.init('shared/XGBoost-integration',
             api_token='ANONYMOUS')

# Prepare data for XGBoost training

boston = load_boston()
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data['PRICE'] = boston.target
X, y = data.iloc[:,:-1], data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=102030)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Prepare params

params = {'max_depth': 5,
          'eta': 0.5,
          'gamma': 0.1,
          'subsample': 1,
          'lambda': 1,
          'alpha': 0.35,
          'objective': 'reg:squarederror',
          'eval_metric': ['mae', 'rmse']}
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 20

# Train model using `xgb.train()`

neptune.create_experiment(name='xgb', tags=['train'], params=params)
xgb.train(params, dtrain, num_round, watchlist,
          callbacks=[neptune_callback()])

neptune.stop()

# Train model using `xgb.cv()`

neptune.create_experiment(name='xgb', tags=['cv'], params=params)
xgb.cv(params, dtrain, num_boost_round=num_round, nfold=7,
       callbacks=[neptune_callback()])

neptune.stop()

# Train model using `sklearn` API

neptune.create_experiment(name='xgb', tags=['sklearn'], params=params)
reg = xgb.XGBRegressor(**params)
reg.fit(X_train, y_train,
        eval_metric=['mae', 'rmse'],
        eval_set=[(X_test, y_test)],
        callbacks=[neptune_callback()])

neptune.stop()