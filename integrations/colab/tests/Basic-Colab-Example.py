# Quickstart

# Step 1: Install ```neptune-client```

pip install neptune-client

# Step 2: Initialize Neptune

## Initialize a public project

import neptune

neptune.init(project_qualified_name='shared/onboarding', api_token='ANONYMOUS')

## Initialize a private project

### Step 2(a): Get your Neptune API token
# 
# ![image](https://neptune.ai/wp-content/uploads/get_token.gif)

### Step 2(b): Set the ```api_token``` to initialize Neptune

from getpass import getpass

api_token = getpass('Enter your private Neptune API token: ')

### Step 2(c): Initialize your private project

neptune.init(project_qualified_name='YOUR_USERNAME/sandbox', api_token=api_token)

# Step 3: Create an experiment

neptune.create_experiment()

# Step 4: Log metrics during training

import numpy as np
from time import sleep

neptune.log_metric('single_metric', 0.62)

for i in range(100):
    sleep(0.2) # to see logging live
    neptune.log_metric('random_training_metric', i * np.random.random())
    neptune.log_metric('other_random_training_metric', 0.5 * i * np.random.random())

# tests
current_exp = neptune.get_experiment()

correct_logs = ['single_metric', 'random_training_metric', 'other_random_training_metric']

if set(current_exp.get_logs().keys()) != set(correct_logs):
    raise ValueError()

# Step 5: Stop tracking the experiment