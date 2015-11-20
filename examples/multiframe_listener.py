# coding: utf-8

import numpy as np
import cv2
from pylibfreenect2 import pyFreenect2, pyFrameMap, pySyncMultiFrameListener
from pylibfreenect2 import FrameType

fn = pyFreenect2()
device = fn.openDefaultDevice()

listener = pySyncMultiFrameListener(FrameType.Color | FrameType.Ir | FrameType.Depth)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

device.start()

while True:
    frames = pyFrameMap()
    listener.waitForNewFrame(frames)

    color = frames[FrameType.Color].udata()
    ir = frames[FrameType.Ir].data()
    depth = frames[FrameType.Depth].data()

    cv2.imshow("ir", ir / 65535.)
    cv2.imshow("depth", depth / 4500.)
    cv2.imshow("color", color)

    listener.release(frames)

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break

device.stop()
device.close()
