import os
from subprocess import call
import yaml

with open('config.yaml') as file:
    config = yaml.load(file)

DIRS = config['run_docs']['dirs']
EXCLUDED_PATHS = config['run_docs']['excluded_dirs']
DOCS_PATHS = config['run_docs']['docs_dirs']

RUN_NOTEBOOK = """ipython {}"""
RUN_SCRIPT = """python {}"""

for dir in DIRS:
    for root, dirs, files in os.walk(dir):
        for name in files:
            if any(p in root for p in EXCLUDED_PATHS):
                continue

            if any(p in root for p in DOCS_PATHS) and name.endswith(('.ipynb')):
                path = os.path.join(root, name)

                if name.endswith(('.ipynb')):
                    try:
                        command = RUN_NOTEBOOK.format(path)
                        call(command, shell=True)
                    except Exception as e:
                        print('Running example {} failed with Exception {}'.format(path, e))

                elif name.endswith(('.py')):
                    try:
                        command = RUN_SCRIPT.format(path)
                        call(command, shell=True)
                    except Exception as e:
                        print('Running example {} failed with Exception {}'.format(path, e))