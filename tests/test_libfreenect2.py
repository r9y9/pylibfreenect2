# coding: utf-8

import numpy as np

from pylibfreenect2 import pyFreenect2, pyFrameMap, pySyncMultiFrameListener


import cv2


def test_sync_multi_frame():
    fn = pyFreenect2()
    device = fn.openDefaultDevice()

    listener = pySyncMultiFrameListener()

    # Register listeners
    device.setColorFrameListener(listener)
    device.setIrAndDepthFrameListener(listener)

    device.start()

    m = pyFrameMap()
    listener.waitForNewFrame(m)

    color = m.get("color")
    ir = m.get("ir")
    depth = m.get("depth")

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

    # TODO
    assert ir.data().shape == (ir.width, ir.height)
    assert depth.data().shape == (depth.width, depth.height)
    assert color.udata().shape == (color.width, color.height)

    listener.release(m)

    device.stop()
    device.close()
