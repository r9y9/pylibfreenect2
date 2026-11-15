Developer Documentation
=======================

Run tests
---------

Install the test dependencies and run the unit tests with ``pytest``:

.. code::

    pip install -e '.[test]'
    pytest -v


If you don't have a Kinect v2 connected, skip the tests that need a device:

.. code::

    pytest -v -m 'not require_device'
