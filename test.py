# coding: utf-8

import numpy as np

from pylibfreenect2 import pyFreenect2, pyFrameMap, pySyncMultiFrameListener


import cv2
import matplotlib.pylab as plt


fn = pyFreenect2()
device = fn.openDefaultDevice()

listener = pySyncMultiFrameListener()

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

device.start()

while True:
    m = pyFrameMap()
    listener.waitForNewFrame(m)

    color = m.get("color").udata()
    ir = m.get("ir").data()
    depth = m.get("depth").data()

    cv2.imshow("ir", ir / 65535.)
    cv2.imshow("depth", depth / 4500.)
    cv2.imshow("color", color)

    listener.release(m)

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break

device.stop()
device.close()
