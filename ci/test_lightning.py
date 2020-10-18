from subprocess import call

from .create import build

build('integrations/pytorch-lightning/Neptune-PyTorch-Lightning-basic.ipynb')
build('integrations/pytorch-lightning/Neptune-PyTorch-Lightning-advanced.ipynb')

call('ipython integrations/pytorch-lightning/tests/Neptune-PyTorch-Lightning-basic.py', shell=True)
call('ipython integrations/pytorch-lightning/tests/Neptune-PyTorch-Lightning-basic_upgraded_libs.py', shell=True)
call('ipython integrations/pytorch-lightning/tests/Neptune-PyTorch-Lightning-advanced.py', shell=True)
call('ipython integrations/pytorch-lightning/tests/Neptune-PyTorch-Lightning-advanced_upgraded_libs.py', shell=True)
