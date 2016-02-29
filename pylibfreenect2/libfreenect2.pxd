# distutils: language = c++

"""
libfreenect2 cdef externs

.. note:
    To resolve naming conflicts between C++ and python, I decided here to
    use C name specification with underscore prefix (e.g. `_Frame` for C++
    and `Frame` for python). Separating namespaces is probably better than
    this.
"""

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

    cdef cppclass _Frame "libfreenect2::Frame":
        uint32_t timestamp
        uint32_t sequence
        size_t width
        size_t height
        size_t bytes_per_pixel
        unsigned char* data
        float exposure
        float gain
        float gamma

        _Frame(size_t width, size_t height, size_t bytes_per_pixel, unsigned char*) except +

    cdef cppclass _FrameListener "libfreenect2::FrameListener":
        bool onNewFrame(int, _Frame*)

cdef extern from "frame_listener_impl.h" namespace "libfreenect2":
    cdef cppclass _SyncMultiFrameListener "libfreenect2::SyncMultiFrameListener":
        _SyncMultiFrameListener(unsigned int)

        bool hasNewFrame()
        void waitForNewFrame(map[LibFreenect2FrameType, _Frame*]&)
        void release(map[LibFreenect2FrameType, _Frame*]&)

cdef extern from "libfreenect2.hpp" namespace "libfreenect2":
    cdef cppclass _Freenect2Device "libfreenect2::Freenect2Device":
        unsigned int VendorId
        unsigned int ProductId
        unsigned int ProductIdPreview

        string getSerialNumber()
        string getFirmwareVersion()

        cppclass _ColorCameraParams "ColorCameraParams":
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

        cppclass _IrCameraParams "IrCameraParams":
            float fx, fy, cx, cy, k1, k2, k3, p1, p2

        _ColorCameraParams getColorCameraParams()
        _IrCameraParams getIrCameraParams()

        # void setColorCameraParams(ColorCameraParams &)
        # void setIrCameraParams(const Freenect2Device::IrCameraParams &)

        void setColorFrameListener(_FrameListener*)
        void setIrAndDepthFrameListener(_FrameListener*)

        void start()
        void stop()
        void close()

cdef extern from "registration.h" namespace "libfreenect2":
    cdef cppclass _Registration "libfreenect2::Registration":
        _Registration(_Freenect2Device._IrCameraParams, _Freenect2Device._ColorCameraParams) except +

        # undistort/register a whole image
        void apply(const _Frame*, const _Frame*, _Frame*, _Frame*, const bool, _Frame*) const

cdef extern from "packet_pipeline.h" namespace "libfreenect2":
    cdef cppclass _PacketPipeline "libfreenect2::PacketPipeline":
        _PacketPipeline *getRgbPacketParser() const
        _PacketPipeline *getIrPacketParser() const

    cdef cppclass _CpuPacketPipeline "libfreenect2::CpuPacketPipeline":
        _CpuPacketPipeline()

    cdef cppclass _OpenGLPacketPipeline "libfreenect2::OpenGLPacketPipeline":
        _OpenGLPacketPipeline(void*, bool)

    cdef cppclass _OpenCLPacketPipeline "libfreenect2::OpenCLPacketPipeline":
        _OpenCLPacketPipeline(const int)


cdef extern from "libfreenect2.hpp" namespace "libfreenect2":
    cdef cppclass _Freenect2 "libfreenect2::Freenect2":
        _Freenect2() except +

        int enumerateDevices()

        string getDeviceSerialNumber(int)
        string getDefaultDeviceSerialNumber()

        _Freenect2Device *openDevice(int)
        _Freenect2Device *openDevice(int idx, const _PacketPipeline *)
        _Freenect2Device *openDevice(const string &)
        _Freenect2Device *openDevice(const string &, const _PacketPipeline *)

        _Freenect2Device *openDefaultDevice()
        _Freenect2Device *openDefaultDevice(const _PacketPipeline *)
