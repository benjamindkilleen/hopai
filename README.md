# Hop AI Tutorial

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](<https://opensource.org/licenses/MIT>)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

PyTorch template project with simple `click`-based CLI. Includes a simple training loop, logging,
and a few other utilities.  For building a more complex project, I *highly* recommend using
[Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template). This is a highly
mature, similarly structured approach to building a PyTorch project, with detailed best practices.
The purpose of this library is to provide a simple template for small projects, for educational
purposes, while demonstrating some best practices for larger projects.

## Installation

These instructions assume a working installation of [Anaconda](https://www.anaconda.com/).

```bash
git clone git@github.com:benjamindkilleen/hopai.git
cd hopai
conda env create -f environment.yml
```

Depending on your desired configuration, you may need to install the
[PyTorch](https://pytorch.org/get-started/locally/) package separately. This can be done following
the instructions on the PyTorch website, in an empty conda environment. Then you can install the
remaining packages with:

```bash
conda activate hopai
pip install -r requirements.txt
pip install -e .
```

This is only necessary if the installation from `environment.yml` fails.

## Usage

```bash
python main.py train
```

## License

This project is licensed under the terms of the MIT license.
