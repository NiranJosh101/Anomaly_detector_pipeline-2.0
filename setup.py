from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str)-> List[str]:
    """This function will return the list of requirements from the given file path."""
    requirements = []
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements=[req.replace('\n', '') for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
name='Anomaly Detection Pipeline',
version='0.1.0',
author='Josh',
author_email='niranjosh011@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt'),
)