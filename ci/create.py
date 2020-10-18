import os
from subprocess import call

import yaml

from ci.utils import clean_py_script

with open('config.yaml') as file:
    config = yaml.load(file)

FILES_TO_CREATE = config['create_docs_paths']

def nbconvert(**kwargs):
    command = """jupyter nbconvert \
        --TagRemovePreprocessor.enabled=True \
        --TagRemovePreprocessor.remove_cell_tags="{tags}" \
        --output-dir {output_dir} \
        --to {format} {notebook_filename}
    """.format(**kwargs)
    call(command, shell=True)

def build(path):
    root_dir = os.path.dirname(path)
    docs_dir = os.path.join(root_dir, 'docs')
    tests_dir = os.path.join(root_dir, 'tests')
    showcase_dir = os.path.join(root_dir, 'showcase')

    for dir_path in [docs_dir, tests_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # create notebook without tests
    nbconvert(
        tags=repr(['tests', 'library_updates']),
        output_dir=showcase_dir,
        format="notebook",
        notebook_filename=path
    )

    # create notebook without tests and comments
    nbconvert(
        tags=repr(['comment','tests','library_updates','exclude']),
        output_dir=docs_dir,
        format="notebook",
        notebook_filename=path
    )

    # create .py script
    nbconvert(
        tags=repr(['comment','installation', 'neptune_stop','tests','library_updates','bash_code','exclude']),
        output_dir=docs_dir,
        format="python",
        notebook_filename=path
    )
    filename = os.path.basename(path)
    path_of_py_file = os.path.join(docs_dir, filename).replace('.ipynb', '.py')
    clean_py_script(path_of_py_file)

    # create .py ipython script -> you need to run it with ipython my_file.py
    nbconvert(
        tags=repr(['comment','neptune_stop', 'library_updates','bash_code','exclude']),
        output_dir=tests_dir,
        format="python",
        notebook_filename=path
    )
    filename = os.path.basename(path)
    path_of_py_file = os.path.join(tests_dir, filename).replace('.ipynb', '.py')
    clean_py_script(path_of_py_file)

    # create .py ipython script with upgraded libraries-> you need to run it with ipython my_file.py
    path_upgraded_libs = path.replace('.ipynb', '_upgraded_libs.ipynb')
    call('cp {} {}'.format(path, path_upgraded_libs), shell=True)

    nbconvert(
        tags=repr(['comment','neptune_stop','bash_code','exclude']),
        output_dir=tests_dir,
        format="python",
        notebook_filename=path_upgraded_libs
    )
    filename = os.path.basename(path_upgraded_libs)
    path_of_py_file = os.path.join(tests_dir, filename).replace('.ipynb', '.py')
    clean_py_script(path_of_py_file)

    call('rm {}'.format(path_upgraded_libs), shell=True)

if __name__ == "__main__":
    for path in FILES_TO_CREATE:
        build(path)
