# # XGBoost integration
# Here, we present XGBoost integration with Neptune that lets you automatically log metrics (train, eval), save trained model to Neptune and much more.
# 
# Usage is easy: just pass `neptune_callback` to training function like any other xgboost callback.
# 
# To try integration simply run this Notebook top to bottom. It works outside-the-box :)
# 
# You can log multiple data types:
# 
#     * Log metrics (train and eval) after each boosting iteration.
#     * Log model (Booster) to Neptune after last boosting iteration.
#     * Log feature importance to Neptune as image after last boosting iteration.
#     * Log visualized trees to Neptune as images after last boosting iteration.

# Resources
# * [Tutorial](https://docs.neptune.ai/integrations/xgboost.html?utm_source=colab&utm_medium=notebook&utm_campaign=integration-xgboost),
# * [Implementation on GitHub](https://github.com/neptune-ai/neptune-contrib/blob/master/neptunecontrib/monitoring/xgboost_monitor.py),
# * [Reference documentation](https://neptune-contrib.readthedocs.io/user_guide/monitoring/xgboost.html).

# Visual overview
# ![xgboost-integration-tour](https://raw.githubusercontent.com/neptune-ai/neptune-colab-examples/master/_static/xgboost-tour.gif "XGBoost integration tour")

# Install dependencies
# This demo requires few Python libs. Let's install them.

get_ipython().system("pip install 'neptune-contrib[monitoring]>=0.18.4'")

import neptune
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# here you import `neptune_calback` that does the magic (the open source magic :)
from neptunecontrib.monitoring.xgboost_monitor import neptune_callback

# Set project
# For this demonstration, I use public user: `neptuner`, who has `ANONYMOUS` token .
# 
# Thanks to this you can run this code as is and see results in Neptune :)

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
          'silent': 1,
          'subsample': 1,
          'lambda': 1,
          'alpha': 0.35,
          'objective': 'reg:linear',
          'eval_metric': ['mae', 'rmse']}
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 20

# Train model using `xgb.train()`

# Example experiment: [https://ui.neptune.ml/shared/XGBoost-integration/e/XGB-41](https://ui.neptune.ai/shared/XGBoost-integration/e/XGB-41?utm_source=colab&utm_medium=notebook&utm_campaign=integration-xgboost)

neptune.create_experiment(name='xgb', tags=['train'], params=params)
xgb.train(params, dtrain, num_round, watchlist,
          callbacks=[neptune_callback(log_tree=[0,1,2])])

neptune.stop()

# Train model using `xgb.cv()`

# Example experiment: [https://ui.neptune.ml/shared/XGBoost-integration/e/XGB-42](https://ui.neptune.ai/shared/XGBoost-integration/e/XGB-42?utm_source=colab&utm_medium=notebook&utm_campaign=integration-xgboost)

neptune.create_experiment(name='xgb', tags=['cv'], params=params)
xgb.cv(params, dtrain, num_boost_round=num_round, nfold=7,
       callbacks=[neptune_callback(log_tree=[0, 1, 2, 3, 4])])

neptune.stop()

# Train model using `sklearn` API

# Example experiment: [https://ui.neptune.ml/shared/XGBoost-integration/e/XGB-43](https://ui.neptune.ai/shared/XGBoost-integration/e/XGB-43?utm_source=colab&utm_medium=notebook&utm_campaign=integration-xgboost)

neptune.create_experiment(name='xgb', tags=['sklearn'], params=params)
reg = xgb.XGBRegressor(**params)
reg.fit(X_train, y_train,
        eval_metric=['mae', 'rmse'],
        eval_set=[(X_test, y_test)],
        callbacks=[neptune_callback(log_tree=[0,1])])

neptune.stop()

# # Did you like it?

# If so, feel free to try it on your data.
# 
## [Register here](https://neptune.ai/register)