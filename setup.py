from setuptools import setup, find_packages

def _require_from_file(file):
    return open(file).read().splitlines()

setup(
    name='python_modules',
    version='0.1',
    url='https://github.com/krocky-cooky/python_modules',
    author='krocky-cooky',
    packages = find_packages()
)