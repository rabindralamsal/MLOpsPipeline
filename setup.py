from setuptools import find_packages, setup
from typing import List

def requirements(file_path: str)-> List[str]:
    '''
    Returns list of requirements from a requirements file.
    '''
    req = []
    with open(file_path, 'r') as f:
        for line in f:
            req.append(line.strip())
    return req[:-1]

setup(
    name='MLOpsPipeline',
    version='0.0.1',
    author='Rabindra Lamsal',
    author_email="rabindralamsal@outlook.com",
    packages=find_packages(),
    install_requires=requirements('requirements.txt'),
)