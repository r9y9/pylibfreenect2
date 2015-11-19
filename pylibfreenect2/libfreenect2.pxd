# distutils: language = c++

from libc.stdint cimport uint32_t
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.map cimport map

cdef enum FrameType:
    Color = 1
    Ir = 2
    Depth = 4

cdef extern from "frame_listener.hpp" namespace "libfreenect2":
    cdef cppclass Frame:
        uint32_t timestamp
        uint32_t sequence
        size_t width
        size_t height
        size_t bytes_per_pixel
        unsigned char* data

        Frame(size_t width, size_t height, size_t bytes_per_pixel) except +

    cdef cppclass FrameListener:
        bool onNewFrame(int, Frame*)

cdef extern from "frame_listener_impl.h" namespace "libfreenect2":
    cdef cppclass SyncMultiFrameListener:
        SyncMultiFrameListener(unsigned int)

        bool hasNewFrame()
        void waitForNewFrame(map[int, Frame*]&)
        void release(map[int, Frame*]&)


cdef extern from "libfreenect2.hpp" namespace "libfreenect2":
    cdef cppclass Freenect2Device:
        unsigned int VendorId
        unsigned int ProductId
        unsigned int ProductIdPreview

        string getSerialNumber()
        string getFirmwareVersion()

        #ColorCameraParams getColorCameraParams()
        #IrCameraParams getIrCameraParams()
        #void setColorCameraParams(ColorCameraParams &)
        # virtual void setIrCameraParams(const Freenect2Device::IrCameraParams &params) = 0;

        void setColorFrameListener(FrameListener*)
        void setIrAndDepthFrameListener(FrameListener*)

        void start()
        void stop()
        void close()

cdef extern from "libfreenect2.hpp" namespace "libfreenect2":
    cdef cppclass Freenect2:
        Freenect2() except +

        int enumerateDevices()

        string getDeviceSerialNumber(int)
        string getDefaultDeviceSerialNumber()

        Freenect2Device *openDevice(int);
        # Freenect2Device *openDevice(int idx, const PacketPipeline *factory);
        Freenect2Device *openDevice(const string &)
        # Freenect2Device *openDevice(const string &serial, const PacketPipeline *factory);

        Freenect2Device *openDefaultDevice()
        #Freenect2Device *openDefaultDevice(const PacketPipeline *factory);
