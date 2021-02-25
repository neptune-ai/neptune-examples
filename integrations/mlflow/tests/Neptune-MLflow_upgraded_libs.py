# Neptune + MLflow

# Before you start

## Install Dependencies

get_ipython().system(' pip install --quiet mlflow==1.12.1 neptune-mlflow==0.2.5 neptune-client==0.4.132')

get_ipython().system(' pip install --quiet --upgrade mlflow neptune-mlflow neptune-client')

## Create some MLflow runs

import os
from random import random, randint
import mlflow 

# start a run
mlflow.start_run()

# Log a parameter (key-value pair)
mlflow.log_param("param1", randint(0, 100))

# Log a metric; metrics can be updated throughout the run
mlflow.log_metric("foo", random())
mlflow.log_metric("foo", random()+1)
mlflow.log_metric("foo", random()+2)
mlflow.log_metric("foo", random()+3)

mlflow.log_metric("bar", random())
mlflow.log_metric("bar", random()+1)
mlflow.log_metric("bar", random()+2)
mlflow.log_metric("bar", random()+3)

# Log an artifact (output file)
os.makedirs("outputs", exist_ok=True)
with open("outputs/test.txt", "w") as f:
    f.write("hello world!")
mlflow.log_artifacts("outputs")

mlflow.end_run()

# Step 1: Set your Environment Variables

get_ipython().run_line_magic('env', 'NEPTUNE_API_TOKEN=ANONYMOUS')
get_ipython().run_line_magic('env', 'NEPTUNE_PROJECT=shared/mlflow-integration')

# Step 2: Sync your MLruns with Neptune

get_ipython().system(' neptune mlflow')

# **Note:**  
# You can specify the path to the directory where the 'mlruns' directory is. 

# See converted experiments
# Click on the link(s) above to browse the MLflow run in Neptune or go to [shared/mlflow-integration project](https://ui.neptune.ai/o/shared/org/mlflow-integration/experiments?viewId=7608998d-4828-48c5-81cc-fb9ec625e206).