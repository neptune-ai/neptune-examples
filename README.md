# neptune-examples

Examples for using Neptune to keep track of your experiments.

You can run every example with zero setup as an "ANONYMOUS" Neptune user (no registration needed).

## Quickstarts

- Use Neptune API to log your first experiment [code](./quickstarst/first-experiment/docs/Use-Neptune-API-to-log-your-first-experiment.py) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/quickstarts/first-experiment/Use-Neptune-API-to-log-your-first-experiment.ipynb)
- Monitor ML runs live [code](./quickstarst/monitor-ml-runs/Monitor-ML-runs-live.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/quickstarst/monitor-ml-runs/Monitor-ML-runs-live.ipynb)
- Organize ML experiments [code](./quickstarst/organize-ml-experimentation/Organize-ML-experiments.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/quickstarts/organize-ml-experimentation/Organize-ML-experiments.ipynb)

## Product Tours

- Neptune API Tour [code](./product-tours/how-it-works/Neptune-API-Tour.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/product-tours/how-it-works/Neptune-API-Tour.ipynb)

## Integrations

- PyTorch Lightning
    - Basic [code](./integrations/pytorch-lightning/Neptune-PyTorch-Ligthning-Basic.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/integrations/pytorch-lightning/Neptune-PyTorch-Ligthning-basic.ipynb)
    - Advanced [code](./integrations/pytorch-lightning/Neptune-PyTorch-Ligthning-Advanced.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/integrations/pytorch-lightning/Neptune-PyTorch-Ligthning-advanced.ipynb)
- XGBoost [code](./integrations/xgboost/Neptune-XGBoost.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/integrations/xgboost/Neptune-XGBoost.ipynb)
- R [code](./integrations/r/Neptune-R.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neptune-ai/neptune-examples/blob/master/integrations/r/Neptune-R.ipynb)

## Notes

The following files are a temporary fix (leaving old paths for now):
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


Some of the colabs are linked to from our [documentation](https://docs.neptune.ai) and they are tested. 
If you are working on one of those remember to add tags:
- 'comment' 
- 'tests'
- 'header'
- 'code'
- 'installation'
- 'library_updates'
- 'neptune_stop'

Those will automatically remove certain cells from the scripts or colab notebooks used in the docs and tests.