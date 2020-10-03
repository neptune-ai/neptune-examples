import re
from pathlib import Path


def clean_py_script(filename):
    EXCLUDED_STR = ['# In[', '#!/usr', '# coding:']
    filename = Path.cwd() / filename
    code_text = filename.read_text().split('\n')
    lines = [line for line in code_text if all(l not in line for l in EXCLUDED_STR)]
    lines = [line.replace('# ##', '#') for line in lines]
    clean_code = '\n'.join(lines)
    clean_code = re.sub(r'\n{2,}', '\n\n', clean_code)
    filename.write_text(clean_code.strip())
