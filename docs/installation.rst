Installation
============

Supported platforms
-------------------

This package is tested with the following platforms:

- Mac OS X 10.11.4 with clang (Apple LLVM version 7.3.0) (locally)
- Ubuntu 14.04 with gcc 4.8.4 (Travis CI)
- Windows with Visual Studio 2015 x64 (AppVeyor CI, only for python 3.5)

Others might be supported, but not guaranteed yet.

Build requirements
------------------

- `libfreenect2 <https://github.com/OpenKinect/libfreenect2>`_ (0.2.0 or later)
- python (2.7, 3.4 or 3.5)
- numpy
- cython
- C++ compiler (clang, gcc or MSVC)

Building pylibfreenect2
-----------------------

The package requires libfreenect2 installed and its location in advance. Please make sure that:

- You have installed `libfreenect2 <https://github.com/OpenKinect/libfreenect2>`_ correctly and confirm that Protonect works.
- You have set ``LIBFREENECT2_INSTALL_PREFIX`` environmental variable (default: ``/usr/local/``) and confirm that ``${LIBFREENECT2_INSTALL_PREFIX}include/libfreenect2/config.h`` exists.


After that, you should be able to install pylibfreenect2 as follows:

.. code::

    pip install pylibfreenect2

If you want the development version of pylibfreenect2,

.. code::

    pip install git+https://github.com/r9y9/pylibfreenect2

or clone the repository and then:

.. code::

    python setup.py install

This should resolve the package dependencies and install ``pylibfreenect2`` property.

.. note::
    If you have installed libfreenect2 in your custom path (e.g. ~/freenect2),
    make sure that the libfreenect2 shared library is in your library search
    path (``DYLD_LIBRARY_PATH`` for osx and ``LD_LIBRARY_PATH`` for linux).
