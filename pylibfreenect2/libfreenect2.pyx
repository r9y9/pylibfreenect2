# coding: utf-8
# cython: boundscheck=True, wraparound=True

"""
libfreenect2 wrapper implementation

.. note:
    For consistency, all exported classes and methods are designed to have
    same name in `libfreenect2`:
    e.g: `Freenect2`, not `pyFreenect2`
    e.g. `getDeviceSerialNumber`, not `get_device_serial_number`
"""

import numpy as np

cimport numpy as np
np.import_array()

cimport cython
cimport libfreenect2

from pylibfreenect2 import FrameType

cdef class Frame:
    cdef _Frame* ptr
    cdef bool take_ownership
    cdef int frame_type

    def __cinit__(self, width=None, height=None, bytes_per_pixel=None, int frame_type=-1):
        w,h,b = width, height, bytes_per_pixel
        all_none = (w is None) and (h is None) and (b is None)
        all_not_none = (w is not None) and (h is not None) and (b is not None)
        assert all_none or all_not_none

        self.frame_type = frame_type

        if all_not_none:
            self.take_ownership = True
            self.ptr = new _Frame(width, height, bytes_per_pixel, NULL)
        else:
            self.take_ownership = False

    def __dealloc__(self):
        if self.take_ownership and self.ptr is not NULL:
            del self.ptr

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

    @property
    def bytes_per_pixel(self):
        return self.ptr.bytes_per_pixel

    @property
    def exposure(self):
        return self.ptr.exposure

    @property
    def gain(self):
        return self.ptr.gain

    @property
    def gamma(self):
        return self.ptr.gamma

    cdef __uint8_data(self):
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.ptr.height
        shape[1] = <np.npy_intp> self.ptr.width
        shape[2] = <np.npy_intp> 4
        cdef np.ndarray array = np.PyArray_SimpleNewFromData(3, shape, np.NPY_UINT8,
         self.ptr.data)

        return array

    cdef __float32_data(self):
        cdef np.npy_intp shape[2]
        shape[0] = <np.npy_intp> self.ptr.height
        shape[1] = <np.npy_intp> self.ptr.width
        cdef np.ndarray array = np.PyArray_SimpleNewFromData(2, shape, np.NPY_FLOAT32,
         self.ptr.data)

        return array

    def asarray(self):
        if self.frame_type < 0:
            raise ValueError("Cannnot determine type of raw data. Use astype instead.")

        if self.frame_type == FrameType.Color:
            return self.astype(np.uint8)
        elif self.frame_type == FrameType.Ir or self.frame_type == FrameType.Depth:
            return self.astype(np.float32)
        else:
            assert False

    def astype(self, data_type):
        if data_type != np.uint8 and data_type != np.float32:
            raise ValueError("np.uint8 or np.float32 is only supported")
        if data_type == np.uint8:
            return self.__uint8_data()
        else:
            return self.__float32_data()


cdef class FrameListener:
    cdef _FrameListener* listener_ptr_alias


cdef intenum_to_frame_type(int n):
    if n == FrameType.Color:
        return Color
    elif n == FrameType.Ir:
        return Ir
    elif n == FrameType.Depth:
        return Depth
    else:
        raise ValueError("Not supported")

cdef str_to_int_frame_type(str s):
    s = s.lower()
    if s == "color":
        return FrameType.Color
    elif s == "ir":
        return FrameType.Ir
    elif s == "depth":
        return FrameType.Depth
    else:
        raise ValueError("Not supported")

cdef str_to_frame_type(str s):
    return intenum_to_frame_type(str_to_int_frame_type(s))


cdef class FrameMap:
    cdef map[LibFreenect2FrameType, _Frame*] internal_frame_map
    cdef bool take_ownership

    def __cinit__(self, bool take_ownership=False):
        self.take_ownership = take_ownership

    def __dealloc__(self):
        # Since libfreenect2 is for now designed to release FrameMap explicitly,
        # __dealloc__  do nothing by default (take_ownership = False)
        if self.take_ownership:
            # similar to SyncMultiFrameListener::release(FrameMap &frame)
            # do nothing if already released
            for key in self.internal_frame_map:
                if key.second != NULL:
                    del key.second
                    key.second = NULL

    def __getitem__(self, key):
        cdef LibFreenect2FrameType frame_type
        cdef intkey

        if isinstance(key, int) or isinstance(key, FrameType):
            frame_type = intenum_to_frame_type(key)
            intkey = key
        elif isinstance(key, str):
            frame_type = str_to_frame_type(key)
            intkey = str_to_int_frame_type(key)
        else:
            raise KeyError("")

        cdef _Frame* frame_ptr = self.internal_frame_map[frame_type]
        cdef Frame frame = Frame(frame_type=intkey)
        frame.ptr = frame_ptr
        return frame


cdef class SyncMultiFrameListener(FrameListener):
    cdef _SyncMultiFrameListener* ptr

    def __cinit__(self, unsigned int frame_types=<unsigned int>(
                        FrameType.Color | FrameType.Ir | FrameType.Depth)):
        self.ptr = new _SyncMultiFrameListener(frame_types)
        self.listener_ptr_alias = <_FrameListener*> self.ptr

    def __dealloc__(self):
        if self.ptr is not NULL:
            del self.ptr

    def hasNewFrame(self):
        return self.ptr.hasNewFrame()

    def waitForNewFrame(self):
        cdef FrameMap frame_map = FrameMap(take_ownership=False)
        self.ptr.waitForNewFrame(frame_map.internal_frame_map)
        return frame_map


    def release(self, FrameMap frame_map):
        self.ptr.release(frame_map.internal_frame_map)


cdef class ColorCameraParams:
    cdef _Freenect2Device._ColorCameraParams params

    # TODO: wrap all instance variables
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


cdef class IrCameraParams:
    cdef _Freenect2Device._IrCameraParams params

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

cdef class Registration:
    cdef _Registration* ptr

    def __cinit__(self, IrCameraParams irparams, ColorCameraParams cparams):
        cdef _Freenect2Device._IrCameraParams i = irparams.params
        cdef _Freenect2Device._ColorCameraParams c = cparams.params
        self.ptr = new _Registration(i, c)

    def __dealloc__(self):
        if self.ptr is not NULL:
            del self.ptr

    def apply(self, Frame color, Frame depth, Frame undistored, Frame registered, enable_filter=True, Frame bigdepth=None):
        assert color.take_ownership == False
        assert depth.take_ownership == False
        assert undistored.take_ownership == True
        assert registered.take_ownership == True
        assert bigdepth is None or bigdepth.take_ownership == True

        cdef _Frame* bigdepth_ptr = <_Frame*>(NULL) if bigdepth is None else bigdepth.ptr

        self.ptr.apply(color.ptr, depth.ptr, undistored.ptr, registered.ptr, enable_filter, bigdepth_ptr)


# MUST be declared before backend specific includes
cdef class PacketPipeline:
    cdef _PacketPipeline* pipeline_ptr_alias

    # NOTE: once device is opened with pipeline, pipeline will be
    # releaseed in the destructor of Freenect2DeviceImpl
    cdef bool owned_by_device


cdef class CpuPacketPipeline(PacketPipeline):
    cdef _CpuPacketPipeline* pipeline

    def __cinit__(self):
        self.pipeline = new _CpuPacketPipeline()
        self.pipeline_ptr_alias = <_PacketPipeline*>self.pipeline
        self.owned_by_device = False

    def __dealloc__(self):
        if not self.owned_by_device:
            if self.pipeline is not NULL:
                del self.pipeline

IF LIBFREENECT2_WITH_OPENGL_SUPPORT == True:
    include "opengl_packet_pipeline.pxi"

IF LIBFREENECT2_WITH_OPENCL_SUPPORT == True:
    include "opencl_packet_pipeline.pxi"


cdef class Freenect2Device:
    cdef _Freenect2Device* ptr

    def getSerialNumber(self):
        return self.ptr.getSerialNumber()

    def getFirmwareVersion(self):
        return self.ptr.getFirmwareVersion()

    def getColorCameraParams(self):
        cdef _Freenect2Device._ColorCameraParams params
        params = self.ptr.getColorCameraParams()
        cdef ColorCameraParams pyparams = ColorCameraParams()
        pyparams.params = params
        return pyparams

    def getIrCameraParams(self):
        cdef _Freenect2Device._IrCameraParams params
        params = self.ptr.getIrCameraParams()
        cdef IrCameraParams pyparams = IrCameraParams()
        pyparams.params = params
        return pyparams

    def setColorFrameListener(self, FrameListener listener):
        self.ptr.setColorFrameListener(listener.listener_ptr_alias)

    def setIrAndDepthFrameListener(self, FrameListener listener):
        self.ptr.setIrAndDepthFrameListener(listener.listener_ptr_alias)

    def start(self):
        self.ptr.start()

    def stop(self):
        self.ptr.stop()

    def close(self):
        self.ptr.close()


cdef class Freenect2:
    cdef _Freenect2* ptr

    def __cinit__(self):
        self.ptr = new _Freenect2();

    def __dealloc__(self):
        if self.ptr is not NULL:
            del self.ptr

    def enumerateDevices(self):
        return self.ptr.enumerateDevices()

    def getDeviceSerialNumber(self, int idx):
        return self.ptr.getDeviceSerialNumber(idx)

    def getDefaultDeviceSerialNumber(self):
        return self.ptr.getDefaultDeviceSerialNumber()

    cdef __openDevice__intidx(self, int idx, PacketPipeline pipeline):
        cdef _Freenect2Device* dev_ptr
        if pipeline is None:
            dev_ptr = self.ptr.openDevice(idx)
        else:
            dev_ptr = self.ptr.openDevice(idx, pipeline.pipeline_ptr_alias)
            pipeline.owned_by_device = True

        cdef Freenect2Device device = Freenect2Device()
        device.ptr = dev_ptr
        return device

    cdef __openDevice__stridx(self, string serial, PacketPipeline pipeline):
        cdef _Freenect2Device* dev_ptr
        if pipeline is None:
            dev_ptr = self.ptr.openDevice(serial)
        else:
            dev_ptr = self.ptr.openDevice(serial, pipeline.pipeline_ptr_alias)
            pipeline.owned_by_device = True

        cdef Freenect2Device device = Freenect2Device()
        device.ptr = dev_ptr
        return device

    def openDevice(self, name, PacketPipeline pipeline=None):
        if isinstance(name, int):
            return self.__openDevice__intidx(name, pipeline)
        elif isinstance(name, str) or isinstance(name, bytes):
            return self.__openDevice__stridx(name, pipeline)
        else:
            raise ValueError("device name must be of str, bytes or integer type")

    def openDefaultDevice(self, PacketPipeline pipeline=None):
        cdef _Freenect2Device* dev_ptr

        if pipeline is None:
            dev_ptr = self.ptr.openDefaultDevice()
        else:
            dev_ptr = self.ptr.openDefaultDevice(pipeline.pipeline_ptr_alias)
            pipeline.owned_by_device = True

        cdef Freenect2Device device = Freenect2Device()
        device.ptr = dev_ptr
        return device
