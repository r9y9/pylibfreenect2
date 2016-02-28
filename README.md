# pylibfreenect2

[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)

A python interface for [libfreenect2](https://github.com/OpenKinect/libfreenect2). The package is compatible with python 2.7-3.5.

Note that this is much work in progress. Currently only tested on osx 10.10.4.

## Build requirements

- [libfreenect2](https://github.com/OpenKinect/libfreenect2) 0.1.0 or later
- python 2.7 or later
- numpy
- cython

## Installation

The package requires libfreenect2 installed in your system path, so please make sure that you have installed [libfreenect2](https://github.com/OpenKinect/libfreenect2) correctly. If you have confirmed that Protonect is working for you, then you should be able to install pylibfreenect2 as follows:


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

## Documentation

Please check [the official doc](https://openkinect.github.io/libfreenect2/). APIs are basically same between C++ and python but actually slightly different. pylibfreenect2 API docs will be comming soon.  


### How it works

Similar to Protonect example in libfreenect2, an example for grabbing and visualizing Kinect v2 data is available:

```
python examples/multiframe_listener.py
```

It requires a python binding of opencv for visualization. If you are using [anaconda](https://www.continuum.io/), you can install it by:

```
conda install opencv
```
