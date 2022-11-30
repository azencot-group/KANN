import os
from setuptools import setup, find_packages


global __version__
__version__ = None

with open('kann/version.py') as f:
  exec(f.read(), globals())

setup(
    name='kann',
    version=__version__,
    description='Koopman Analysis of Neural Networks (KANN).',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy >=1.12',
        'torch',
        'matplotlib',
        'sklearn',
        'seaborn',
        'argparse',
        'arff2pandas',
        'tfds-nightly',
        'tensorflow_text',
        'tensorflow',
        'tqdm'
    ]
)
