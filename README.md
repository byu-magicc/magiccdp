# MAGICC Lab Differentiable Programming Tutorials

## Installation Instructions

* Create a [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) environment with Python 3.12

```bash
mamba create -n magiccdp python=3.12
```

* Activate the Mamba environment

```bash
mamba activate magiccdp
```

* `cd` into the `magiccdp` folder and `pip install` the package

```bash
pip install -e .
```

(**Note:** The `-e` option stands for "editable". It allows you to change the code in the folder and run those changes immediately. Don't leave this out!)

* Run the notebook using `marimo edit <notebook-name>`. For example,

```bash
marimo edit marimo_tutorials/big_picture.py
```

### Alternative to get Python 3.12 on Ubuntu 22.04

```console
$ sudo add-apt-repository ppa:deadsnakes/ppa
$ sudo apt update
$ sudo apt install python3.12-dev python3.12-venv
$ python3.12 -m venv ~/path/to/where/you/want/the/venv/created
$ source ~/path/to/where/you/want/the/venv/created/bin/activate
$ cd ~/path/to/this/cloned/repo/magiccdp
$ pip install -e .
```

Continue as above.

### Alternative for a native Py3.12 environment

e.g. For MacOS with up-to-date Python

```console
$ python -m venv ~/path/to/where/you/want/the/venv/created
$ source ~/path/to/where/you/want/the/venv/created/bin/activate
$ cd ~/path/to/this/cloned/repo/magiccdp
$ pip install -e .
```

continue as above
