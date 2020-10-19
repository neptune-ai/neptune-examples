from subprocess import call
from pathlib import Path

from .build import build_tests

source_files = [
    'integrations/pytorch-lightning/Neptune-PyTorch-Lightning-basic.ipynb',
    'integrations/pytorch-lightning/Neptune-PyTorch-Lightning-advanced.ipynb'
]

for filename in source_files:
    build_tests(Path(filename))

test_files = [
    'integrations/pytorch-lightning/tests/Neptune-PyTorch-Lightning-basic.py',
    'integrations/pytorch-lightning/tests/Neptune-PyTorch-Lightning-basic_upgraded_libs.py',
    'integrations/pytorch-lightning/tests/Neptune-PyTorch-Lightning-advanced.py',
    'integrations/pytorch-lightning/tests/Neptune-PyTorch-Lightning-advanced_upgraded_libs.py'
]

for filename in test_files:
    retcode = call('ipython ' + filename, shell=True)
    if retcode:
        raise Exception('Test {} failed'.format(filename))
