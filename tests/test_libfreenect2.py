# coding: utf-8

import numpy as np

from pylibfreenect2 import pyFreenect2, pyFrameMap, pySyncMultiFrameListener
from pylibfreenect2 import FrameType


def test_sync_multi_frame():
    fn = pyFreenect2()
    device = fn.openDefaultDevice()

    listener = pySyncMultiFrameListener(
        FrameType.Color | FrameType.Ir | FrameType.Depth)

    # Register listeners
    device.setColorFrameListener(listener)
    device.setIrAndDepthFrameListener(listener)

    device.start()

    frames = pyFrameMap()
    listener.waitForNewFrame(frames)

    color = frames[FrameType.Color]
    ir = frames[FrameType.Ir]
    depth = frames[FrameType.Depth]

    ### Color ###
    assert color.width == 1920
    assert color.height == 1080
    assert color.bytes_per_pixel == 4

    assert ir.width == 512
    assert ir.height == 424
    assert ir.bytes_per_pixel == 4

    assert depth.width == 512
    assert depth.height == 424
    assert depth.bytes_per_pixel == 4

    assert color.udata().shape == (color.height, color.width, 4)
    assert ir.data().shape == (ir.height, ir.width)
    assert depth.data().shape == (depth.height, depth.width)

    listener.release(frames)

    device.stop()
    device.close()
