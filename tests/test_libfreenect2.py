# coding: utf-8

import numpy as np
import sys

from nose.tools import raises
from nose.plugins.attrib import attr

from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame, FrameMap
from pylibfreenect2 import (Logger,
                            createConsoleLogger,
                            createConsoleLoggerWithDefaultLevel,
                            getGlobalLogger,
                            setGlobalLogger,
                            LoggerLevel)


def test_frame():
    frame = Frame(512, 424, 4)
    assert frame.width == 512
    assert frame.height == 424
    assert frame.bytes_per_pixel == 4
    assert frame.exposure == 0
    assert frame.gain == 0
    assert frame.gamma == 0


def test_logger():
    logger_default = createConsoleLoggerWithDefaultLevel()
    assert isinstance(logger_default, Logger)

    for level in [LoggerLevel.NONE, LoggerLevel.Error,
                  LoggerLevel.Warning,
                  LoggerLevel.Info,
                  LoggerLevel.Debug]:
        logger = createConsoleLogger(level)
        setGlobalLogger(logger)
        assert getGlobalLogger().level() == level

    # Turn logging off
    setGlobalLogger(None)
    # Set to default
    setGlobalLogger(logger_default)

    logger = getGlobalLogger()
    if sys.version_info.major >= 3:
        message = b"test debugging message"
    else:
        message = "test debugging message"
    logger.log(LoggerLevel.Debug, message)


def test_enumerateDevices():
    fn = Freenect2()
    fn.enumerateDevices()


@attr('require_device')
def test_openDefaultDevice():
    fn = Freenect2()

    num_devices = fn.enumerateDevices()
    assert num_devices > 0

    device = fn.openDefaultDevice()

    device.stop()
    device.close()


@attr('require_device')
def test_startStreams():
    def __test(enable_rgb, enable_depth):
        fn = Freenect2()
        num_devices = fn.enumerateDevices()
        assert num_devices > 0
        device = fn.openDefaultDevice()

        types = 0
        if enable_rgb:
            types |= FrameType.Color
        if enable_depth:
            types |= (FrameType.Ir | FrameType.Depth)
        listener = SyncMultiFrameListener(types)

        device.setColorFrameListener(listener)
        device.setIrAndDepthFrameListener(listener)

        device.startStreams(rgb=enable_rgb, depth=enable_depth)
        # test if we can get one frame at least
        frames = listener.waitForNewFrame()
        listener.release(frames)

        device.stop()
        device.close()

    __test(True, False)
    __test(False, True)


@attr('require_device')
def test_sync_multi_frame():
    fn = Freenect2()

    num_devices = fn.enumerateDevices()
    assert num_devices > 0

    serial = fn.getDefaultDeviceSerialNumber()
    assert serial == fn.getDeviceSerialNumber(0)

    device = fn.openDevice(serial)

    assert fn.getDefaultDeviceSerialNumber() == device.getSerialNumber()
    device.getFirmwareVersion()

    listener = SyncMultiFrameListener(
        FrameType.Color | FrameType.Ir | FrameType.Depth)

    # Register listeners
    device.setColorFrameListener(listener)
    device.setIrAndDepthFrameListener(listener)

    device.start()

    # Registration
    registration = Registration(device.getIrCameraParams(),
                                device.getColorCameraParams())
    undistorted = Frame(512, 424, 4)
    registered = Frame(512, 424, 4)

    # optional parameters for registration
    bigdepth = Frame(1920, 1082, 4)
    color_depth_map = np.zeros((424, 512), np.int32)

    # test if we can get two frames at least
    frames = listener.waitForNewFrame()
    listener.release(frames)

    # frames as a first argment also should work
    frames = FrameMap()
    listener.waitForNewFrame(frames)

    color = frames[FrameType.Color]
    ir = frames[FrameType.Ir]
    depth = frames[FrameType.Depth]

    for frame in [ir, depth]:
        assert frame.exposure == 0
        assert frame.gain == 0
        assert frame.gamma == 0

    for frame in [color]:
        assert frame.exposure > 0
        assert frame.gain > 0
        assert frame.gamma > 0

    registration.apply(color, depth, undistorted, registered)

    # with optinal parameters
    registration.apply(color, depth, undistorted, registered,
                       bigdepth=bigdepth,
                       color_depth_map=color_depth_map.ravel())

    registration.undistortDepth(depth, undistorted)

    assert color.width == 1920
    assert color.height == 1080
    assert color.bytes_per_pixel == 4

    assert ir.width == 512
    assert ir.height == 424
    assert ir.bytes_per_pixel == 4

    assert depth.width == 512
    assert depth.height == 424
    assert depth.bytes_per_pixel == 4

    assert color.asarray().shape == (color.height, color.width, 4)
    assert ir.asarray().shape == (ir.height, ir.width)
    assert depth.asarray(np.float32).shape == (depth.height, depth.width)

    listener.release(frames)

    def __test_cannot_determine_type_of_frame(frame):
        frame.asarray()

    for frame in [registered, undistorted]:
        yield raises(ValueError)(__test_cannot_determine_type_of_frame), frame

    # getPointXYZ
    x, y, z = registration.getPointXYZ(undistorted, 512 // 2, 424 // 2)
    if not np.isnan([x, y, z]).any():
        assert z > 0

    # getPointXYZRGB
    x, y, z, b, g, r = registration.getPointXYZRGB(undistorted, registered,
                                                   512 // 2, 424 // 2)
    if not np.isnan([x, y, z]).any():
        assert z > 0
    assert np.isfinite([b, g, r]).all()

    for pix in [b, g, r]:
        assert pix >= 0 and pix <= 255

    device.stop()
    device.close()
