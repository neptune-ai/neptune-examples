# neptune-examples

Examples for using Neptune to keep track of your experiments.

You can run every example with zero setup as an "ANONYMOUS" Neptune user (no registration needed).

For each example you can:
- see the script (code), 
- see rendered notebook (notebook) 
- or open the notebook in Colab (open in colab)

## Quick Starts

- Use Neptune API to log your first experiment [code](./quick-starts/first-experiment/docs/Use-Neptune-API-to-log-your-first-experiment.py) [notebook](./quick-starts/first-experiment/showcase/Use-Neptune-API-to-log-your-first-experiment.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/quick-starts/first-experiment/showcase/Use-Neptune-API-to-log-your-first-experiment.ipynb)
- Monitor ML runs live [code](./quick-starts/monitor-ml-runs/docs/Monitor-ML-runs-live.py) [notebook](./quick-starts/monitor-ml-runs/showcase/Monitor-ML-runs-live.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/quick-starts/monitor-ml-runs/showcase/Monitor-ML-runs-live.ipynb)
- Organize ML experiments [code](./quick-starts/organize-ml-experimentation/docs/Organize-ML-experiments.py) [notebook](./quick-starts/organize-ml-experimentation/showcase/Organize-ML-experiments.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/quick-starts/organize-ml-experimentation/showcase/Organize-ML-experiments.ipynb)

## Product Tours

- Neptune API Tour [code](./product-tours/how-it-works/docs/Neptune-API-Tour.py) [notebook](./product-tours/how-it-works/showcase/Neptune-API-Tour.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/product-tours/how-it-works/showcase/Neptune-API-Tour.ipynb)

## Integrations

- PyTorch [code](./integrations/pytorch/docs/Neptune-PyTorch.py) [notebook](./integrations/pytorch/showcase/Neptune-PyTorch.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/integrations/pytorch/showcase/Neptune-PyTorch.ipynb)
- Tensorflow / Keras [code](./integrations/tensorflow-keras/docs/Neptune-TensorFlow-Keras.py) [notebook](./integrations/tensorflow-keras/showcase/Neptune-TensorFlow-Keras.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/integrations/tensorflow-keras/showcase/Neptune-TensorFlow-Keras.ipynb)
- PyTorch Lightning
    - Basic [code](./integrations/pytorch-lightning/docs/Neptune-PyTorch-Lightning-basic.py) [notebook](./integrations/pytorch-lightning/showcase/Neptune-PyTorch-Lightning-basic.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/integrations/pytorch-lightning/showcase/Neptune-PyTorch-Lightning-basic.ipynb)
    - Advanced [code](./integrations/pytorch-lightning/docs/Neptune-PyTorch-Lightning-advanced.py) [notebook](./integrations/pytorch-lightning/showcase/Neptune-PyTorch-Lightning-advanced.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/integrations/pytorch-lightning/showcase/Neptune-PyTorch-Lightning-advanced.ipynb)
- XGBoost [code](./integrations/xgboost/docs/Neptune-XGBoost.py) [notebook](./integrations/xgboost/showcase/Neptune-XGBoost.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/integrations/xgboost/showcase/Neptune-XGBoost.ipynb)
- Optuna [code](./integrations/optuna/docs/Neptune-Optuna.py) [notebook](./integrations/optuna/showcase/Neptune-Optuna.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/integrations/optuna/showcase/Neptune-Optuna.ipynb)
- TensorBoard [code](./integrations/tensorboard/docs/Neptune-TensorBoard.py) [notebook](./integrations/tensorboard/showcase/Neptune-TensorBoard.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/integrations/tensorboard/showcase/Neptune-TensorBoard.ipynb)
- MLflow [code](./integrations/mlflow/docs/Neptune-MLflow.py) [notebook](./integrations/mlflow/showcase/Neptune-MLflow.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/integrations/mlflow/showcase/Neptune-MLflow.ipynb)
- R [notebook](./integrations/r/Neptune-R.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/integrations/r/Neptune-R.ipynb)
- Google Colab [notebook](./integrations/showcase/Basic-Colab-Example.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/integrations/colab/showcase/Basic-Colab-Example.ipynb)

## Contributing

### Required sections

When you create an example notebook remember to create the following sections:
- Installation: all the libraries needed to run this in Colab in their current versions
- Library updates: all the libraries from the point before but with `--upgrade` at the end
- Neptune stop: after each experiment you should stop the experiment

### Tags

Each cell should be tagged with one or many of the following tags:
- `comment`
- `tests`
- `header`
- `code`
- `installation`
- `library_updates`
- `neptune_stop`

### Creating scripts, notebooks and tests

For each example notebook you create the following artifacts will be created automatically:

- `*/docs/NOTEBOOK_NAME.ipynb` : it excluded cells tagged with 'comment', 'tests', 'library_updates'
- `*/docs/NOTEBOOK_NAME.py` : it excluded cells tagged with'comment', 'tests', 'library_updates', 'neptune_stop'
- `*/showcase/NOTEBOOK_NAME.ipynb` : it excluded cells tagged with 'tests', 'library_updates'
- `*/tests/NOTEBOOK_NAME.py` : it excluded cells tagged with 'comment', 'library_updates' , 'neptune_stop'
- `*/tests/NOTEBOOK_NAME_upgraded_libs.py` : it excluded cells tagged with 'comment', 'neptune_stop'

To do it run:

```bash
python ci/build.py
```

To run all of those scripts execute:

```baash
python ci/run.py
```

Under the hood all the examples are run with `ipython SCRIPT_NAME.py` to run the library installations from within the script.

You can also run both with:

```bash
source run.sh
```

### Configuration

You can configure which notebooks should be created or run with the `config.yaml` configuration file. 

**create_docs_paths**

This is used in `ci/create.py`.
Pass paths to all the notebook for which you want to create artifacts.
For example:

```yaml
create_docs_paths: [
  'integrations/pytorch-lightning/Neptune-PyTorch-Lightning-basic.ipynb',
  'quick-starts/monitor-ml-runs/Monitor-ML-runs-live.ipynb',
  'quick-starts/organize-ml-experimentation/Organize-ML-experiments.ipynb',
]
```

**run_docs_paths**

This is used in `ci/run.py`.

*included_patterns*
   
Pass patterns that you want to include. By default those are `/docs`, `/tests` and `/showcase`
For example:

```yaml
  included_patterns: [
    '/docs',
    '/tests',
    '/showcase',
  ]
```

  *included_patterns*
   
Pass patterns that you want to exclude. By default those are `/.ipynb_checkpoints`, `/.git` and `/showcase`
For example:

```yaml
  excluded_patterns: [
    '/.ipynb_checkpoints',
    '/.git',
  ]
```

### Notes

- The following files are a temporary fix (leaving old paths for now):
    - logging_snippets.ipynb      
    - Neptune-API-Tour.ipynb  
    - Organize-ML-experiments.ipynb        
    - r-integration.ipynb  
    - Use-Neptune-API-to-log-your-first-experiment.ipynb
    - Monitor-ML-runs-live.ipynb  
    - neptune_test_run.ipynb  
    - pytorch_lightning-integration.ipynb  
    - Template.ipynb       
    - xgboost-integration.ipynb
- XGBoost integration (upgraded libs case) is not tested on Win with Python 3.8, as it causes [tkinter error](https://github.com/neptune-ai/neptune-examples/runs/1309037471?check_suite_focus=true).
- "Neptune-API-Tour" (product-tours) is not tested on Windows, because unable to install Tensorflow on a Windows CI server ([error msg](https://github.com/neptune-ai/neptune-examples/pull/17/checks?check_run_id=1308563484#step:10:328)).
