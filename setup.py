#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='project',
    version='0.1.0',
    description='Implementation of deep image hashing methods',
    author='Guru Swaroop Bennabhaktula',
    author_email='bguruswaroop@gmail.com',
    url='https://github.com/bgswaroop/deep-hashing',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)
