# MAGICC Lab Differentiable Programming Tutorials

## Installation Instructions

* Create a [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) environment with Python 12

```bash
mamba create -n magiccdp python=3.12
```

* Activate the Mamba environment

```bash
mamba activate magiccdp
```

* `cd` into the `magiccdp` folder and pip install the package

```bash
pip install -e .
```

(**Note:** The `-e` option stands for "editable". It allows you to change the code in the folder and run those changes immediately. Don't leave this out!)

* Run the notebook using `marimo edit <notebook-name>`. For example,

```bash
marimo edit marimo_tutorials/big_picture.py
```