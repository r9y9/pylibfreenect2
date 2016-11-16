Developer Documentation
=======================

Run tests
---------

Assuming that you have installed ``nose``, you can run unittests by:

.. code::

    nosetests -v -w tests/


If you don't have kinect v2 connected, you can run unittests that doesn't require devices by:

.. code::

    nosetests -v -w tests/ -a '!require_device'

This is exactly what CI does for automatic testing.
