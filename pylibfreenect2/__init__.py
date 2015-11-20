# coding: utf-8

"""
A python wrapper for `libfreenect2 <https://github.com/OpenKinect/libfreenect2>`_.
"""

from __future__ import division, print_function, absolute_import

import pkg_resources

__version__ = pkg_resources.get_distribution('pylibfreenect2').version

from enum import IntEnum


class FrameType(IntEnum):
    """libfreenect::Frame::Type
    """
    Color = 1
    Ir = 2
    Depth = 4


from .libfreenect2 import *
