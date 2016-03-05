# pylibfreenect2

[![Build Status](https://travis-ci.org/r9y9/pylibfreenect2.svg?branch=master)](https://travis-ci.org/r9y9/pylibfreenect2)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)

A python interface for [libfreenect2](https://github.com/OpenKinect/libfreenect2). The package is compatible with python 2.7-3.5.

## Supported platforms

- Mac OS X
- Ubuntu 14.04

## Documentation

See http://r9y9.github.io/pylibfreenect2/ for Python API docs.

## Build requirements

- [libfreenect2](https://github.com/OpenKinect/libfreenect2) 0.1.0 or later
- python 2.7 or later ([anaconda](https://www.continuum.io/) is recommended)
- numpy
- cython

## Installation

The package requires libfreenect2 installed and the installed location in advance. Please make sure that:

- You have installed [libfreenect2](https://github.com/OpenKinect/libfreenect2) correctly and confirm that Protonect works.
- You have set `LIBFREENECT2_INSTALL_PREFIX` environmental variable (default: `/usr/local/`).

After that, you should be able to install pylibfreenect2 as follows:

```
pip install git+https://github.com/r9y9/pylibfreenect2
```

or clone the repository and then:

```
python setup.py install
```

## Run tests

On top of the project directroy, you can run unittests by:

```
nosetests -v -w tests/
```

It is assumed that you have `nose` installed.


### Getting started

See [examples/multiframe_listener.py](examples/multiframe_listener.py)

The example code demonstrates basic usage of pylibfreenect2. It requires a python binding of opencv for visualization. If you are using [anaconda](https://www.continuum.io/), you can install it by:

```
conda install opencv
```

Have fun!
