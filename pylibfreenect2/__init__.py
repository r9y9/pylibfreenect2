# coding: utf-8

"""
A python interface for `libfreenect2 <https://github.com/OpenKinect/libfreenect2>`_.
The package is compatible with python 2.7-3.5.

https://github.com/r9y9/pylibfreenect2
"""

from __future__ import division, print_function, absolute_import

import pkg_resources

__version__ = pkg_resources.get_distribution('pylibfreenect2').version

from enum import IntEnum


class FrameType(IntEnum):
    """Python-side enum for ``libfreenect::Frame::Type`` in C++.

    The value can be Color, Ir or Depth.

    .. warning::
        The name is slightly different between Python and C++.

    Suppose the following C++ code:

    .. code-block:: c++

        libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
        libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
        libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];

    This can be tranlated in Python like:

    .. code-block:: python

        rgb = frames[pylibfreenect2.FrameType.Color]
        ir = frames[pylibfreenect2.FrameType.Ir]
        depth = frames[pylibfreenect2.FrameType.Depth]

    or you can use str key:

    .. code-block:: python

        rgb = frames["color"]
        ir = frames["ir"]
        depth = frames["depth"]

    See also
    --------

    pylibfreenect2.Frame
    pylibfreenect2.FrameMap
    pylibfreenect2.SyncMultiFrameListener

    """
    Color = 1
    Ir = 2
    Depth = 4


from .libfreenect2 import *
