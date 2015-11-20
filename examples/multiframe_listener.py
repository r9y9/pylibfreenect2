# coding: utf-8

import numpy as np
import cv2
from pylibfreenect2 import pyFreenect2, pyFrameMap, pySyncMultiFrameListener
from pylibfreenect2 import FrameType, pyRegistration, pyFrame


fn = pyFreenect2()
device = fn.openDefaultDevice()

listener = pySyncMultiFrameListener(
    FrameType.Color | FrameType.Ir | FrameType.Depth)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

device.start()

# NOTE: must be called after device.start()
registration = pyRegistration(device.getIrCameraParams(),
                              device.getColorCameraParams())

undistorted = pyFrame(512, 424, 4)
registered = pyFrame(512, 424, 4)

while True:
    frames = pyFrameMap()
    listener.waitForNewFrame(frames)

    color = frames["color"]
    ir = frames["ir"]
    depth = frames["depth"]

    registration.apply(color, depth, undistorted, registered)

    cv2.imshow("ir", ir.data() / 65535.)
    cv2.imshow("depth", depth.data() / 4500.)
    cv2.imshow("color", color.udata())
    cv2.imshow("registered", registered.udata())

    listener.release(frames)

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break

device.stop()
device.close()
