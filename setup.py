#!/usr/bin/env python
from setuptools import find_packages
from setuptools import setup

setup(
    name="hopai",
    version="0.0.0",
    description="PyTorch training loop template.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Benjamin D. Killeen",
    author_email="killeen@jhu.edu",
    url="https://github.com/benjamindkilleen/hopai",
    install_requires=[
        "click",
        "rich",
        "numpy",
        "torch",
        "torchvision",
    ],
    packages=find_packages(),
)
