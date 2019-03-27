# distutils: language = c++

"""
libfreenect2 cdef externs
"""

from libc.stdint cimport uint32_t
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.map cimport map


cdef extern from "libfreenect2/frame_listener.hpp" namespace "libfreenect2":
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
        float exposure
        float gain
        float gamma

        Frame(size_t width, size_t height, size_t bytes_per_pixel, unsigned char*) except +

    cdef cppclass FrameListener:
        bool onNewFrame(int, Frame*)

cdef extern from "libfreenect2/frame_listener_impl.h" namespace "libfreenect2":
    cdef cppclass SyncMultiFrameListener:
        SyncMultiFrameListener(unsigned int)

        bool hasNewFrame()
        void waitForNewFrame(map[LibFreenect2FrameType, Frame*]&) nogil
        void release(map[LibFreenect2FrameType, Frame*]&)

cdef extern from "libfreenect2/libfreenect2.hpp" namespace "libfreenect2":
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
            float fx, fy, cx, cy, k1, k2, k3, p1, p2

        ColorCameraParams getColorCameraParams()
        IrCameraParams getIrCameraParams()

        # void setColorCameraParams(ColorCameraParams &)
        # void setIrCameraParams(const Freenect2Device::IrCameraParams &)

        void setColorFrameListener(FrameListener*)
        void setIrAndDepthFrameListener(FrameListener*)

        void start()
        bool startStreams(bool, bool)
        void stop()
        void close()

cdef extern from "libfreenect2/registration.h" namespace "libfreenect2":
    cdef cppclass Registration:
        Registration(Freenect2Device.IrCameraParams, Freenect2Device.ColorCameraParams) except +

        # undistort/register a whole image
        void apply(const Frame*, const Frame*, Frame*, Frame*, const bool, Frame*, int*) const

        void undistortDepth(const Frame*, Frame*)

        # construct a 3-D point with color in a point cloud
        void getPointXYZRGB(const Frame*, const Frame*, int, int, float&, float&, float&, float&) const

        # construct a 3-D point in a point cloud
        void getPointXYZ(const Frame*, int, int, float&, float&, float&) const


cdef extern from "libfreenect2/packet_pipeline.h" namespace "libfreenect2":
    cdef cppclass PacketPipeline:
        PacketPipeline *getRgbPacketParser() const
        PacketPipeline *getIrPacketParser() const

    cdef cppclass CpuPacketPipeline:
        CpuPacketPipeline()

    cdef cppclass OpenGLPacketPipeline:
        OpenGLPacketPipeline(void*, bool)

    cdef cppclass OpenCLPacketPipeline:
        OpenCLPacketPipeline(const int)

    cdef cppclass OpenCLKdePacketPipeline:
        OpenCLKdePacketPipeline(const int)


cdef extern from "libfreenect2/libfreenect2.hpp" namespace "libfreenect2":
    cdef cppclass Freenect2:
        Freenect2() except +

        int enumerateDevices()

        string getDeviceSerialNumber(int)
        string getDefaultDeviceSerialNumber()

        Freenect2Device *openDevice(int)
        Freenect2Device *openDevice(int idx, const PacketPipeline *)
        Freenect2Device *openDevice(const string &)
        Freenect2Device *openDevice(const string &, const PacketPipeline *)

        Freenect2Device *openDefaultDevice()
        Freenect2Device *openDefaultDevice(const PacketPipeline *)

cdef extern from "libfreenect2/logger.h" namespace "libfreenect2":
    # ugly but works
    cdef enum LoggerLevel "libfreenect2::Logger::Level":
        None "libfreenect2::Logger::Level::None"
        Error "libfreenect2::Logger::Level::Error"
        Warning "libfreenect2::Logger::Level::Warning"
        Info "libfreenect2::Logger::Level::Info"
        Debug "libfreenect2::Logger::Level::Debug"

    cdef cppclass Logger:
        LoggerLevel level() const
        void log(LoggerLevel, const string &)

    Logger* createConsoleLogger(LoggerLevel)
    Logger* createConsoleLoggerWithDefaultLevel()
    Logger* getGlobalLogger()
    void setGlobalLogger(Logger *logger)
