# Step 1: Install ```neptune-client```

# Step 2: Initialize Neptune

## Initialize a public project

import neptune

neptune.init(project_qualified_name='shared/onboarding', api_token='ANONYMOUS')

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

# Step 5: Stop tracking the experiment