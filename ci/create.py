import os
from subprocess import call

import yaml

from ci.utils import clean_py_script

with open('config.yaml') as file:
    config = yaml.load(file)

FILES_TO_CREATE = config['create_docs_paths']

CREATE_SHOWCASE_NOTEBOOK = """jupyter nbconvert \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags="['tests','library_updates']" \
    --output-dir {} \
    --to notebook {}
    """

CREATE_DOCS_NOTEBOOK = """jupyter nbconvert \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags="['comment','tests','library_updates','exclude']" \
    --output-dir {} \
    --to notebook {}
    """

CREATE_DOCS_SCRIPT = """jupyter nbconvert \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags="['comment','installation', 'neptune_stop','tests','library_updates','bash_code','exclude']" \
    --output-dir {} \
    --to python {}
    """

CREATE_TESTS_SCRIPT = """jupyter nbconvert \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags="['comment','neptune_stop', 'library_updates','bash_code','exclude']" \
    --output-dir {} \
    --to python {}
    """

CREATE_TESTS_SCRIPT_UPGRADED_LIBS = """jupyter nbconvert \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags="['comment','neptune_stop','bash_code','exclude']" \
    --output-dir {} \
    --to python {}
    """

if __name__ == "__main__":

    for path in FILES_TO_CREATE:
        root_dir = os.path.dirname(path)
        docs_dir = os.path.join(root_dir, 'docs')
        tests_dir = os.path.join(root_dir, 'tests')
        showcase_dir = os.path.join(root_dir, 'showcase')

        for dir_path in [docs_dir, tests_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # create notebook without tests
        command = CREATE_SHOWCASE_NOTEBOOK.format(showcase_dir, path)
        call(command, shell=True)

        # create notebook without tests and comments
        command = CREATE_DOCS_NOTEBOOK.format(docs_dir, path)
        call(command, shell=True)

        # create .py script
        command = CREATE_DOCS_SCRIPT.format(docs_dir, path)
        call(command, shell=True)
        filename = os.path.basename(path)
        path_of_py_file = os.path.join(docs_dir, filename).replace('.ipynb', '.py')
        clean_py_script(path_of_py_file)

        # create .py ipython script -> you need to run it with ipython my_file.py
        command = CREATE_TESTS_SCRIPT.format(tests_dir, path)
        call(command, shell=True)
        filename = os.path.basename(path)
        path_of_py_file = os.path.join(tests_dir, filename).replace('.ipynb', '.py')
        clean_py_script(path_of_py_file)

        # create .py ipython script with upgraded libraries-> you need to run it with ipython my_file.py
        path_upgraded_libs = path.replace('.ipynb', '_upgraded_libs.ipynb')
        call('cp {} {}'.format(path, path_upgraded_libs), shell=True)

        command = CREATE_TESTS_SCRIPT_UPGRADED_LIBS.format(tests_dir, path_upgraded_libs)
        call(command, shell=True)
        filename = os.path.basename(path_upgraded_libs)
        path_of_py_file = os.path.join(tests_dir, filename).replace('.ipynb', '.py')
        clean_py_script(path_of_py_file)

        call('rm {}'.format(path_upgraded_libs), shell=True)
