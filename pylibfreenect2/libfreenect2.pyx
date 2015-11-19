# coding: utf-8
# cython: boundscheck=True, wraparound=True
import numpy as np

cimport numpy as np
np.import_array()

cimport cython
cimport libfreenect2

cdef class pyFrame:
    cdef Frame* ptr

    @property
    def bytes_per_pixel(self):
        return self.ptr.bytes_per_pixel

    @property
    def timestamp(self):
        return self.ptr.timestamp

    @property
    def sequence(self):
        return self.ptr.sequence

    @property
    def width(self):
        return self.ptr.width

    @property
    def height(self):
        return self.ptr.height

    # TODO: more safe interrface
    def udata(self):
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.ptr.height
        shape[1] = <np.npy_intp> self.ptr.width
        shape[2] = <np.npy_intp> 4
        cdef np.ndarray array = np.PyArray_SimpleNewFromData(3, shape, np.NPY_UINT8,
         self.ptr.data)

        return array

    def data(self):
        cdef np.npy_intp shape[2]
        shape[0] = <np.npy_intp> self.ptr.height
        shape[1] = <np.npy_intp> self.ptr.width
        cdef np.ndarray array = np.PyArray_SimpleNewFromData(2, shape, np.NPY_FLOAT32,
         self.ptr.data)

        return array



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


    def __get(self, FrameType key=Color):
        cdef Frame* frame_ptr = self.internal_frame_map[key]
        frame = pyFrame()
        frame.ptr = frame_ptr
        return frame

    def get(self, type="color"):
        if type == "color":
            return self.__get(Color)
        elif type == "ir":
            return self.__get(Ir)
        elif type == "depth":
            return self.__get(Depth)
        else:
            raise ValueError("invalid frame type")


cdef class pySyncMultiFrameListener(pyFrameListener):
    cdef SyncMultiFrameListener* ptr

    def __cinit__(self, unsigned int frame_types=IColor | IIr | IDepth):
        self.ptr = new SyncMultiFrameListener(frame_types)

    def __dealloc__(self):
        if self.ptr is not NULL:
            del self.ptr

    def hasNewFrame(self):
        return self.ptr.hasNewFrame()

    def waitForNewFrame(self, pyFrameMap frame_map):
        self.ptr.waitForNewFrame(frame_map.internal_frame_map)

    def release(self, pyFrameMap frame_map):
        self.ptr.release(frame_map.internal_frame_map)


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
