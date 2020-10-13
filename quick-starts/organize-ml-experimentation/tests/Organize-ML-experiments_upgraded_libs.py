# Organize ML experiments

# Setup

get_ipython().system(' pip install scikit-learn==0.23.1 joblib==0.15.1 neptune-client==0.4.123')

get_ipython().system(' pip install scikit-learn joblib neptune-client --upgrade')

# Step 1: Create a basic training script

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from joblib import dump

data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
                                                    test_size=0.4, random_state=1234)

params = {'n_estimators': 10,
          'max_depth': 3,
          'min_samples_leaf': 1,
          'min_samples_split': 2,
          'max_features': 3,
          }

clf = RandomForestClassifier(**params)

clf.fit(X_train, y_train)
y_train_pred = clf.predict_proba(X_train)
y_test_pred = clf.predict_proba(X_test)

train_f1 = f1_score(y_train, y_train_pred.argmax(axis=1), average='macro')
test_f1 = f1_score(y_test, y_test_pred.argmax(axis=1), average='macro')
print(f'Train f1:{train_f1} | Test f1:{test_f1}')

# Step 2: Initialize Neptune

import neptune

neptune.init(project_qualified_name='shared/onboarding', # change this to your `workspace_name/project_name`
             api_token='ANONYMOUS', # change this to your api token
            )

# Step 3: Create an experiment and save parameters

neptune.create_experiment(name='great-idea', params=params)

# Step 4. Add tags to organize things

neptune.append_tag(['experiment-organization', 'me'])

# Step 5. Add logging of train and evaluation metrics

neptune.log_metric('train_f1', train_f1)
neptune.log_metric('test_f1', test_f1)

# Step 6. Run a few experiments with different parameters

# tests
current_exp = neptune.get_experiment()

correct_logs = ['train_f1', 'test_f1']

if set(current_exp.get_logs().keys()) != set(correct_logs):
    raise ValueError()