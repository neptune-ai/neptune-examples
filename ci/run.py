import os
from subprocess import call

import yaml

with open('config.yaml') as file:
    config = yaml.load(file)

EXCLUDED_PATTERNS = config['run_docs_paths']['excluded_patterns']
INCLUDED_PATTERNS = config['run_docs_paths']['included_patterns']
RUN_SCRIPT = """ipython {}"""

for root, dirs, files in os.walk('.'):
    for name in files:
        if any(p in root for p in EXCLUDED_PATTERNS) and name.endswith(('.ipynb')):
            continue
        if any(p in root for p in INCLUDED_PATTERNS) and name.endswith(('.ipynb')):
            path = os.path.join(root, name)
            if name.endswith(('.ipynb')) or name.endswith('.py'):
                try:
                    command = RUN_SCRIPT.format(path)
                    call(command, shell=True)
                except Exception as e:
                    with open('error_logs.txt', 'w+') as f:
                        message = 'file {} failed'.format(path)
                        f.write(message)
