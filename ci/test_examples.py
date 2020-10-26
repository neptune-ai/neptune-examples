import os

from glob import glob
from subprocess import check_call

import pytest

test_files = glob('**/tests/*.py', recursive=True)

excluded_files = []
if os.name == 'nt': # if OS is Windows
    # Excluding because unable to install Tensorflow on a Windows CI server with
    #
    # ERROR: Could not install packages due to an EnvironmentError: [WinError 5] Access is denied: 'c:\\hostedtoolcache\\windows\\python\\3.6.8\\x64\\lib\\site-packages\\~umpy\\.libs\\libopenblas.NOIJJG62EMASZI6NYURL6JBKM4EVBGM7.gfortran-win_amd64.dll'
    # Consider using the `--user` option or check the permissions.
    excluded_files.extend(glob('product-tours/how-it-works/tests/*.py', recursive=True))

@pytest.mark.parametrize("filename", [f for f in test_files if f not in excluded_files])
def test_examples(filename):
    check_call('ipython ' + filename, shell=True)
