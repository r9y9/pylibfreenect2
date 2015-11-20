# distutils: language = c++

from libc.stdint cimport uint32_t
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.map cimport map


cdef extern from "frame_listener.hpp" namespace "libfreenect2":
    # ugly but works
    cdef enum LibFreenect2FrameType "libfreenect2::Frame::Type":
        Color "libfreenect2::Frame::Type::Color"
        Ir "libfreenect2::Frame::Type::Ir"
        Depth "libfreenect2::Frame::Type::Depth"

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
        void waitForNewFrame(map[LibFreenect2FrameType, Frame*]&)
        void release(map[LibFreenect2FrameType, Frame*]&)


cdef extern from "libfreenect2.hpp" namespace "libfreenect2":
    cdef cppclass Freenect2Device:
        unsigned int VendorId
        unsigned int ProductId
        unsigned int ProductIdPreview

        string getSerialNumber()
        string getFirmwareVersion()

        cppclass ColorCameraParams:
            float fx, fy, cx, cy

            float shift_d, shift_m

            float mx_x3y0
            float mx_x0y3
            float mx_x2y1
            float mx_x1y2
            float mx_x2y0
            float mx_x0y2
            float mx_x1y1
            float mx_x1y0
            float mx_x0y1
            float mx_x0y0

            float my_x3y0
            float my_x0y3
            float my_x2y1
            float my_x1y2
            float my_x2y0
            float my_x0y2
            float my_x1y1
            float my_x1y0
            float my_x0y1
            float my_x0y0

        cppclass IrCameraParams:
            float fx, fy, cx, cy, k1, k2, k3, p1, p2;

        ColorCameraParams getColorCameraParams()
        IrCameraParams getIrCameraParams()

        # void setColorCameraParams(ColorCameraParams &)
        # void setIrCameraParams(const Freenect2Device::IrCameraParams &)

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
