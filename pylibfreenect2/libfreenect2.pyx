# coding: utf-8
# cython: boundscheck=True, wraparound=True
import numpy as np

cimport numpy as np
cimport cython
cimport libfreenect2

cdef class pyFrame:
    cdef Frame* ptr

    @property
    def bytes_per_pixel(self):
        return self.ptr.bytes_per_pixel

# TODO: utilize inheritance
cdef class pyFrameListener:
    def onNewFrame(self):
        pass


cdef class pyFrameMap:
    cdef map[FrameType, Frame*] internal_frame_map

    def __cinit__(self):
        pass

    def __dealloc__(self):
        pass


cdef class pySyncMultiFrameListener(pyFrameListener):
    cdef SyncMultiFrameListener* ptr

    def __cinit__(self, unsigned int frame_types=Color |Ir | Depth):
        self.ptr = new SyncMultiFrameListener(frame_types)

    def hasNewFrame(self):
        return self.ptr.hasNewFrame()

    def waitForNewFrame(self, pyFrameMap frame_map):
        # TODO
        pass
        # cdef map[FrameType, Frame*] m = frame_map.internal_frame_map;
        # cdef map[int, Frame*] internal_frame_map
        # self.ptr.waitForNewFrame(internal_frame_map)


cdef class pyFreenect2Device:
    cdef Freenect2Device* ptr

    def getSerialNumber(self):
        return self.ptr.getSerialNumber()

    def getFirmwareVersion(self):
        return self.ptr.getFirmwareVersion()

    def setColorFrameListener(self, pySyncMultiFrameListener listener):
        cdef FrameListener* listener_ptr = <FrameListener*>(listener.ptr)
        self.ptr.setColorFrameListener(listener_ptr)

    def setIrAndDepthFrameListener(self, pySyncMultiFrameListener listener):
        cdef FrameListener* listener_ptr = <FrameListener*>(listener.ptr)
        self.ptr.setIrAndDepthFrameListener(listener_ptr)

    def start(self):
        self.ptr.start()

    def stop(self):
        self.ptr.stop()

    def close(self):
        self.ptr.close()


cdef pyFreenect2Device_Init(Freenect2Device* ptr):
    device = pyFreenect2Device()
    device.ptr = ptr
    return device


cdef class pyFreenect2:
    cdef Freenect2* ptr

    def __cinit__(self):
        self.ptr = new Freenect2();

    def __dealloc__(self):
        if self.ptr is not NULL:
            del self.ptr

    def enumerateDevices(self):
        return self.ptr.enumerateDevices()

    def getDeviceSerialNumber(self, int idx):
        return self.ptr.getDeviceSerialNumber(idx)

    def getDefaultDeviceSerialNumber(self):
        cdef string s = self.ptr.getDefaultDeviceSerialNumber()
        #TODO

    def openDevice(self, int idx):
        cdef Freenect2Device* dev_ptr = self.ptr.openDevice(idx)
        return pyFreenect2Device_Init(dev_ptr)

    def openDefaultDevice(self):
        cdef Freenect2Device* dev_ptr = self.ptr.openDefaultDevice()
        return pyFreenect2Device_Init(dev_ptr)
        # return pyFreenect2Device(self.thisptr.openDefaultDevice())
