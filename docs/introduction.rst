Installation guide
==================

Build requirements
------------------

- `libfreenect2 <https://github.com/OpenKinect/libfreenect2>`_ 0.1.0 or later
- python 2.7 or later
- numpy
- cython

Installation
------------

The package requires libfreenect2 installed and its location in advance. Please make sure that:

- You have installed `libfreenect2 <https://github.com/OpenKinect/libfreenect2>`_ correctly and confirm that Protonect works.
- You have set ``LIBFREENECT2_INSTALL_PREFIX`` environmental variable (default: ``/usr/local/``).


After that, you should be able to install pylibfreenect2 as follows:

.. code::

    pip install git+https://github.com/r9y9/pylibfreenect2

or clone the repository and then:

.. code::

    python setup.py install

This should resolve the package dependencies and install ``pylibfreenect2`` property.
