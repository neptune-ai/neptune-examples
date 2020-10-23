from glob import glob
from subprocess import check_call

import pytest

test_files = glob('**/tests/*.py', recursive=True)

excluded_files = []

@pytest.mark.parametrize("filename", [f for f in test_files if f not in excluded_files])
def test_examples(filename):
    check_call('ipython ' + filename, shell=True)
