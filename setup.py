import shutil
from setuptools import setup, find_packages

setup(
    name='runtime_api',
    version = '0.1.0',
    packages=find_packages()
)

shutil.rmtree('build', ignore_errors=True)