import os
from setuptools import setup, find_packages

def get_requirements(filename='requirements.txt'):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires

setup(
    name='jackjack',
    version='0.1.0',
    description='Computer vision',
    author='Minjun Jeon',
    author_email='ghkduadml@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements(),
    setup_requires=['tensorflow'],
)