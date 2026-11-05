Installation
============

Supported platforms
-------------------

The package builds on Linux, macOS and Windows with a C++11 compiler.
It is developed and tested on Linux (Ubuntu, gcc).

Build requirements
------------------

- `libfreenect2 <https://github.com/OpenKinect/libfreenect2>`_ (0.2.0 or later)
- python 3.10 or later
- C++ compiler (clang, gcc or MSVC)

Cython, numpy and setuptools are fetched automatically by ``pip`` at build time.

Locating libfreenect2
---------------------

Build and confirm libfreenect2 works first (e.g. Protonect runs). The build then
looks for libfreenect2 in this order:

1. ``LIBFREENECT2_BUILD_DIR``: an uninstalled CMake build tree, e.g.
   ``libfreenect2/build``. Public headers are taken from
   ``${LIBFREENECT2_SOURCE_DIR}/include`` (default: the build directory's parent),
   generated headers and the library from the build directory.
   No ``cmake --install`` is needed.
2. ``LIBFREENECT2_INSTALL_PREFIX``: an install prefix containing
   ``include/libfreenect2/config.h`` and ``lib/``.
3. ``pkg-config freenect2``.
4. ``/usr/local``.

Building pylibfreenect2
-----------------------

From a clone of the repository:

.. code::

    LIBFREENECT2_BUILD_DIR=/path/to/libfreenect2/build pip install .

or, for development, an editable install:

.. code::

    LIBFREENECT2_BUILD_DIR=/path/to/libfreenect2/build pip install -e . --config-settings editable_mode=compat

``editable_mode=compat`` puts the source tree on ``sys.path`` directly instead
of using an import hook, so static type checkers (Pylance, pyright, mypy) can
find the bundled type stubs. Without it the install still works at runtime,
but editors won't see the types.

The development version can also be installed straight from git:

.. code::

    pip install git+https://github.com/r9y9/pylibfreenect2

Optional dependencies
---------------------

The scripts in ``examples/`` use OpenCV for display, which pylibfreenect2
itself does not depend on. To run them you'll need ``opencv-python``; install
it with the ``examples`` extra:

.. code::

    LIBFREENECT2_BUILD_DIR=/path/to/libfreenect2/build pip install '.[examples]'

The other extras are ``test`` (pytest) and ``docs`` (Sphinx theme and
helpers). To install all of them at once:

.. code::

    LIBFREENECT2_BUILD_DIR=/path/to/libfreenect2/build pip install '.[all]'

.. note::
    On Linux and macOS the libfreenect2 library directory is embedded in the
    extension as an RPATH, so ``LD_LIBRARY_PATH`` / ``DYLD_LIBRARY_PATH`` is not
    needed. The path is absolute: rebuild pylibfreenect2 if you move
    libfreenect2. On Windows, make sure ``freenect2.dll`` is on ``PATH``.
