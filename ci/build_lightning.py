from .build import build

source_files = [
    'integrations/pytorch-lightning/Neptune-PyTorch-Lightning-basic.ipynb',
    'integrations/pytorch-lightning/Neptune-PyTorch-Lightning-advanced.ipynb'
]

if __name__ == '__main__':
    for filename in source_files:
        build(filename)
