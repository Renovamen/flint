from os import path
from setuptools import setup, find_packages

current_path = path.abspath(path.dirname(__file__))

# load content from `README.md`
def readme():
    readme_path = path.join(current_path, 'README.md')
    with open(readme_path, encoding = 'utf-8') as fp:
        return fp.read()

setup(
    name = 'flint',
    version = '0.1.0',
    packages = find_packages(),
    description = 'A toy deep learning framework built with Numpy from scratch with a PyTorch-like API.',
    long_description = readme(),
    long_description_content_type = 'text/markdown',
    license = 'MIT',
    author = 'Xiaohan Zou',
    author_email = 'renovamenzxh@gmail.com',
    url = 'https://github.com/Renovamen/flint'
)
