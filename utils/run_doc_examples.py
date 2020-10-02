import os
from subprocess import call

from utils.postprocess_scripts import clean_py_script

DIRS = ['integrations']
EXCLUDED_PATHS = ['.ipynb_checkpoints','integrations/r']
DOCS_PATHS = ['docs']

RUN_NOTEBOOK = """jupyter nbconvert \
    --to notebook \
    --execute {}
    """

for dir in DIRS:
    for root, dirs, files in os.walk(dir):
        for name in files:
            if any(p in root for p in EXCLUDED_PATHS):
                continue

            if any(p in root for p in DOCS_PATHS) and name.endswith(('.ipynb')):
                path = os.path.join(root, name)

                command = RUN_NOTEBOOK.format(path)
                call(command, shell=True)

