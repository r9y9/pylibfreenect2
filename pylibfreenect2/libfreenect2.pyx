# coding: utf-8
# cython: boundscheck=True, wraparound=True
import numpy as np

cimport numpy as np
np.import_array()

cimport cython
cimport libfreenect2

from pylibfreenect2 import FrameType

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


cdef intenum_to_frame_type(n):
    if n == FrameType.Color:
        return Color
    elif n == FrameType.Ir:
        return Ir
    elif n == FrameType.Depth:
        return Depth
    else:
        raise ValueError("Not supported")

cdef str_to_frame_type(s):
    s = s.lower()
    if s == "color":
        return Color
    elif s == "ir":
        return Ir
    elif s == "depth":
        return Depth
    else:
        raise ValueError("Not supported")

cdef class pyFrameMap:
    cdef map[LibFreenect2FrameType, Frame*] internal_frame_map

    def __cinit__(self):
        pass

    def __dealloc__(self):
        pass

    def __getitem__(self, key):
        cdef LibFreenect2FrameType frame_type
        if isinstance(key, int) or isinstance(key, FrameType):
            frame_type = intenum_to_frame_type(key)
        elif isinstance(key, str):
            frame_type = str_to_frame_type(key)
        else:
            raise KeyError("")

        cdef Frame* frame_ptr = self.internal_frame_map[frame_type]
        frame = pyFrame()
        frame.ptr = frame_ptr
        return frame


cdef class pySyncMultiFrameListener(pyFrameListener):
    cdef SyncMultiFrameListener* ptr

    def __cinit__(self, unsigned int frame_types=<unsigned int>(
                        FrameType.Color | FrameType.Ir | FrameType.Depth)):
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


cdef class pyColorCameraParams:
    cdef Freenect2Device.ColorCameraParams params

    # TODO: non complete. すべて手で書くのは面倒なので、自動生成がいい？
    # pythonでマクロ使えたら楽なんだが、、
    @property
    def fx(self):
        return self.params.fx

    @property
    def fy(self):
        return self.params.fy

    @property
    def cx(self):
        return self.params.cx

    @property
    def cy(self):
        return self.params.cy


cdef class pyIrCameraParams:
    cdef Freenect2Device.IrCameraParams params

    @property
    def fx(self):
        return self.params.fx

    @property
    def fy(self):
        return self.params.fy

    @property
    def cx(self):
        return self.params.cx

    @property
    def cy(self):
        return self.params.cy


cdef class pyFreenect2Device:
    cdef Freenect2Device* ptr

    def getSerialNumber(self):
        return self.ptr.getSerialNumber()

    def getFirmwareVersion(self):
        return self.ptr.getFirmwareVersion()

    def getColorCameraParams(self):
        cdef Freenect2Device.ColorCameraParams params
        params = self.ptr.getColorCameraParams()
        cdef pyColorCameraParams pyparams = pyColorCameraParams()
        pyparams.params = params
        return pyparams

    def getIrCameraParams(self):
        cdef Freenect2Device.IrCameraParams params
        params = self.ptr.getIrCameraParams()
        cdef pyIrCameraParams pyparams = pyIrCameraParams()
        pyparams.params = params
        return pyparams

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
        return self.ptr.getDefaultDeviceSerialNumber()

    cdef __openDevice__intidx(self, int idx):
        cdef Freenect2Device* dev_ptr = self.ptr.openDevice(idx)
        cdef pyFreenect2Device device = pyFreenect2Device()
        device.ptr = dev_ptr
        return device

    cdef __openDevice__stridx(self, string serial):
        cdef Freenect2Device* dev_ptr = self.ptr.openDevice(serial)
        cdef pyFreenect2Device device = pyFreenect2Device()
        device.ptr = dev_ptr
        return device

    def openDevice(self, name):
        if isinstance(name, int):
            return self.__openDevice__intidx(name)
        elif isinstance(name, str):
            return self.__openDevice__stridx(name)
        else:
            ValueError("device name must be str or integer index")

    def openDefaultDevice(self):
        cdef Freenect2Device* dev_ptr = self.ptr.openDefaultDevice()
        cdef pyFreenect2Device device = pyFreenect2Device()
        device.ptr = dev_ptr
        return device
