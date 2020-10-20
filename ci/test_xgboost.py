from subprocess import call
from pathlib import Path

from .build import build_tests

source_files = [
    'integrations/xgboost/Neptune-XGBoost.ipynb'
]

for filename in source_files:
    build_tests(Path(filename))

test_files = [
    'integrations/xgboost/tests/Neptune-XGBoost.py',
    'integrations/xgboost/tests/Neptune-XGBoost_upgraded_libs.py'
]

for filename in test_files:
    retcode = call('ipython ' + filename, shell=True)
    if retcode:
        raise Exception('Test {} failed'.format(filename))
