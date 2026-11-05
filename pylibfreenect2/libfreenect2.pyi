"""Type stubs for the compiled ``pylibfreenect2.libfreenect2`` Cython module.

Hand-written from the module's docstrings and runtime introspection
(pylibfreenect2 0.1.4). Serial numbers / firmware versions are returned as
``bytes`` by the C++ bridge, not ``str``.
"""

from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt

from pylibfreenect2 import FrameType as FrameType
from pylibfreenect2 import LoggerLevel

_FrameKey = FrameType | int | Literal["color", "ir", "depth"]

# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

class Frame:
    """Python interface for ``libfreenect2::Frame``.

    Frames obtained from a ``FrameMap`` do not own their memory and are only
    valid until ``SyncMultiFrameListener.release`` is called. Frames created
    with ``width``/``height``/``bytes_per_pixel`` allocate and own memory.

    C++ source:

    - [`class LIBFREENECT2_API Frame`](file:///home/coder/portland/libfreenect2/include/libfreenect2/frame_listener.hpp#L44) — libfreenect2/include/libfreenect2/frame_listener.hpp:44
    """

    def __init__(
        self,
        width: int | None = None,
        height: int | None = None,
        bytes_per_pixel: int | None = None,
        frame_type: FrameType | int = -1,
        numpy_array: npt.NDArray[Any] | None = None,
    ) -> None:
        """C++ source:

        - [`Frame::Frame(size_t width, size_t height, size_t bytes_per_pixel, unsigned char *data_) :`](file:///home/coder/portland/libfreenect2/src/frame_listener_impl.cpp#L35) — libfreenect2/src/frame_listener_impl.cpp:35
        """
    @property
    def width(self) -> int:
        """C++ source:

        - [`size_t width;           ///< Length of a line (in pixels).`](file:///home/coder/portland/libfreenect2/include/libfreenect2/frame_listener.hpp#L66) — libfreenect2/include/libfreenect2/frame_listener.hpp:66
        """
    @property
    def height(self) -> int:
        """C++ source:

        - [`size_t height;          ///< Number of lines in the frame.`](file:///home/coder/portland/libfreenect2/include/libfreenect2/frame_listener.hpp#L67) — libfreenect2/include/libfreenect2/frame_listener.hpp:67
        """
    @property
    def bytes_per_pixel(self) -> int:
        """C++ source:

        - [`size_t bytes_per_pixel; ///< Number of bytes in a pixel. If frame format is 'Raw' this ...`](file:///home/coder/portland/libfreenect2/include/libfreenect2/frame_listener.hpp#L68) — libfreenect2/include/libfreenect2/frame_listener.hpp:68
        """
    @property
    def timestamp(self) -> int:
        """C++ source:

        - [`uint32_t timestamp;     ///< Unit: 0.125 millisecond. Usually incrementing by 266 (30Hz...`](file:///home/coder/portland/libfreenect2/include/libfreenect2/frame_listener.hpp#L70) — libfreenect2/include/libfreenect2/frame_listener.hpp:70
        """
    @property
    def sequence(self) -> int:
        """C++ source:

        - [`uint32_t sequence;      ///< Increasing frame sequence number`](file:///home/coder/portland/libfreenect2/include/libfreenect2/frame_listener.hpp#L71) — libfreenect2/include/libfreenect2/frame_listener.hpp:71
        """
    @property
    def exposure(self) -> float:
        """C++ source:

        - [`float exposure;         ///< From 0.5 (very bright) to ~60.0 (fully covered)`](file:///home/coder/portland/libfreenect2/include/libfreenect2/frame_listener.hpp#L72) — libfreenect2/include/libfreenect2/frame_listener.hpp:72
        """
    @property
    def gain(self) -> float:
        """C++ source:

        - [`float gain;             ///< From 1.0 (bright) to 1.5 (covered)`](file:///home/coder/portland/libfreenect2/include/libfreenect2/frame_listener.hpp#L73) — libfreenect2/include/libfreenect2/frame_listener.hpp:73
        """
    @property
    def gamma(self) -> float:
        """C++ source:

        - [`float gamma;            ///< From 1.0 (bright) to 6.4 (covered)`](file:///home/coder/portland/libfreenect2/include/libfreenect2/frame_listener.hpp#L74) — libfreenect2/include/libfreenect2/frame_listener.hpp:74
        """
    @overload
    def asarray(self, dtype: type[np.uint8]) -> npt.NDArray[np.uint8]: ...
    @overload
    def asarray(self, dtype: type[np.float32]) -> npt.NDArray[np.float32]: ...
    @overload
    def asarray(self, dtype: None = None) -> npt.NDArray[Any]:
        """Frame to ``numpy.ndarray`` conversion (no copy).

        - Color: ``(height, width, 4)`` ``uint8`` (BGRX)
        - IR / depth: ``(height, width)`` ``float32``

        ``dtype`` is required for frames created without a ``frame_type``
        (e.g. ``Frame(512, 424, 4)`` used with ``Registration.apply``).
        """

class FrameMap:
    """Python interface for ``libfreenect2::FrameMap``.

    Index with a ``FrameType``, its int value, or ``"color"``/``"ir"``/``"depth"``.
    An unknown key raises ``ValueError`` (not ``KeyError``).

    C++ source:

    - [`typedef std::map<Frame::Type, Frame*> FrameMap;`](file:///home/coder/portland/libfreenect2/include/libfreenect2/frame_listener_impl.h#L43) — libfreenect2/include/libfreenect2/frame_listener_impl.h:43
    """

    def __init__(self, take_ownership: bool = False) -> None: ...
    def __getitem__(self, key: _FrameKey) -> Frame: ...

# ---------------------------------------------------------------------------
# Listeners
# ---------------------------------------------------------------------------

class FrameListener:
    """C++ source:

    - [`class LIBFREENECT2_API FrameListener`](file:///home/coder/portland/libfreenect2/include/libfreenect2/frame_listener.hpp#L94) — libfreenect2/include/libfreenect2/frame_listener.hpp:94
    """

class SyncMultiFrameListener(FrameListener):
    """Python interface for ``libfreenect2::SyncMultiFrameListener``.

    ``frame_types`` is a bitwise OR of ``FrameType`` values
    (default: Color | Ir | Depth).

    C++ source:

    - [`class LIBFREENECT2_API SyncMultiFrameListener : public FrameListener`](file:///home/coder/portland/libfreenect2/include/libfreenect2/frame_listener_impl.h#L48) — libfreenect2/include/libfreenect2/frame_listener_impl.h:48
    """

    def __init__(self, frame_types: int = ...) -> None:
        """C++ source:

        - [`SyncMultiFrameListener::SyncMultiFrameListener(unsigned int frame_types) :`](file:///home/coder/portland/libfreenect2/src/frame_listener_impl.cpp#L89) — libfreenect2/src/frame_listener_impl.cpp:89
        """
    def hasNewFrame(self) -> bool:
        """C++ source:

        - [`bool SyncMultiFrameListener::hasNewFrame() const`](file:///home/coder/portland/libfreenect2/src/frame_listener_impl.cpp#L100) — libfreenect2/src/frame_listener_impl.cpp:100
        """
    @overload
    def waitForNewFrame(self, frame_map: FrameMap | None = None) -> FrameMap: ...
    @overload
    def waitForNewFrame(
        self, frame_map: FrameMap | None = None, *, milliseconds: int
    ) -> FrameMap | None:
        """Block until a new frame set arrives.

        With ``milliseconds >= 0``, returns ``None`` on timeout. The returned
        FrameMap must be passed to ``release`` when done.

        C++ source:

        - [`bool SyncMultiFrameListener::waitForNewFrame(FrameMap &frame, int milliseconds)`](file:///home/coder/portland/libfreenect2/src/frame_listener_impl.cpp#L107) — libfreenect2/src/frame_listener_impl.cpp:107
        - [`void SyncMultiFrameListener::waitForNewFrame(FrameMap &frame)`](file:///home/coder/portland/libfreenect2/src/frame_listener_impl.cpp#L132) — libfreenect2/src/frame_listener_impl.cpp:132
        """
    def release(self, frame_map: FrameMap) -> None:
        """C++ source:

        - [`void SyncMultiFrameListener::release(FrameMap &frame)`](file:///home/coder/portland/libfreenect2/src/frame_listener_impl.cpp#L147) — libfreenect2/src/frame_listener_impl.cpp:147
        """

# ---------------------------------------------------------------------------
# Camera parameters & registration
# ---------------------------------------------------------------------------

class ColorCameraParams:
    """Python interface for ``libfreenect2::Freenect2Device::ColorCameraParams``.

    C++ source:

    - [`struct ColorCameraParams`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L58) — libfreenect2/include/libfreenect2/libfreenect2.hpp:58
    """

    fx: float
    """C++ source:

    - [`float fx; ///< Focal length x (pixel)`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L62) — libfreenect2/include/libfreenect2/libfreenect2.hpp:62
    """
    fy: float
    """C++ source:

    - [`float fy; ///< Focal length y (pixel)`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L63) — libfreenect2/include/libfreenect2/libfreenect2.hpp:63
    """
    cx: float
    """C++ source:

    - [`float cx; ///< Principal point x (pixel)`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L64) — libfreenect2/include/libfreenect2/libfreenect2.hpp:64
    """
    cy: float
    """C++ source:

    - [`float cy; ///< Principal point y (pixel)`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L65) — libfreenect2/include/libfreenect2/libfreenect2.hpp:65
    """
    shift_d: float
    """C++ source:

    - [`float shift_d, shift_m;`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L75) — libfreenect2/include/libfreenect2/libfreenect2.hpp:75
    """
    shift_m: float
    """C++ source:

    - [`float shift_d, shift_m;`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L75) — libfreenect2/include/libfreenect2/libfreenect2.hpp:75
    """
    mx_x3y0: float
    """C++ source:

    - [`float mx_x3y0; // xxx`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L77) — libfreenect2/include/libfreenect2/libfreenect2.hpp:77
    """
    mx_x0y3: float
    """C++ source:

    - [`float mx_x0y3; // yyy`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L78) — libfreenect2/include/libfreenect2/libfreenect2.hpp:78
    """
    mx_x2y1: float
    """C++ source:

    - [`float mx_x2y1; // xxy`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L79) — libfreenect2/include/libfreenect2/libfreenect2.hpp:79
    """
    mx_x1y2: float
    """C++ source:

    - [`float mx_x1y2; // yyx`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L80) — libfreenect2/include/libfreenect2/libfreenect2.hpp:80
    """
    mx_x2y0: float
    """C++ source:

    - [`float mx_x2y0; // xx`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L81) — libfreenect2/include/libfreenect2/libfreenect2.hpp:81
    """
    mx_x0y2: float
    """C++ source:

    - [`float mx_x0y2; // yy`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L82) — libfreenect2/include/libfreenect2/libfreenect2.hpp:82
    """
    mx_x1y1: float
    """C++ source:

    - [`float mx_x1y1; // xy`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L83) — libfreenect2/include/libfreenect2/libfreenect2.hpp:83
    """
    mx_x1y0: float
    """C++ source:

    - [`float mx_x1y0; // x`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L84) — libfreenect2/include/libfreenect2/libfreenect2.hpp:84
    """
    mx_x0y1: float
    """C++ source:

    - [`float mx_x0y1; // y`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L85) — libfreenect2/include/libfreenect2/libfreenect2.hpp:85
    """
    mx_x0y0: float
    """C++ source:

    - [`float mx_x0y0; // 1`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L86) — libfreenect2/include/libfreenect2/libfreenect2.hpp:86
    """
    my_x3y0: float
    """C++ source:

    - [`float my_x3y0; // xxx`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L88) — libfreenect2/include/libfreenect2/libfreenect2.hpp:88
    """
    my_x0y3: float
    """C++ source:

    - [`float my_x0y3; // yyy`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L89) — libfreenect2/include/libfreenect2/libfreenect2.hpp:89
    """
    my_x2y1: float
    """C++ source:

    - [`float my_x2y1; // xxy`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L90) — libfreenect2/include/libfreenect2/libfreenect2.hpp:90
    """
    my_x1y2: float
    """C++ source:

    - [`float my_x1y2; // yyx`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L91) — libfreenect2/include/libfreenect2/libfreenect2.hpp:91
    """
    my_x2y0: float
    """C++ source:

    - [`float my_x2y0; // xx`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L92) — libfreenect2/include/libfreenect2/libfreenect2.hpp:92
    """
    my_x0y2: float
    """C++ source:

    - [`float my_x0y2; // yy`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L93) — libfreenect2/include/libfreenect2/libfreenect2.hpp:93
    """
    my_x1y1: float
    """C++ source:

    - [`float my_x1y1; // xy`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L94) — libfreenect2/include/libfreenect2/libfreenect2.hpp:94
    """
    my_x1y0: float
    """C++ source:

    - [`float my_x1y0; // x`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L95) — libfreenect2/include/libfreenect2/libfreenect2.hpp:95
    """
    my_x0y1: float
    """C++ source:

    - [`float my_x0y1; // y`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L96) — libfreenect2/include/libfreenect2/libfreenect2.hpp:96
    """
    my_x0y0: float
    """C++ source:

    - [`float my_x0y0; // 1`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L97) — libfreenect2/include/libfreenect2/libfreenect2.hpp:97
    """
    def __init__(self) -> None: ...

class IrCameraParams:
    """Python interface for ``libfreenect2::Freenect2Device::IrCameraParams``.

    C++ source:

    - [`struct IrCameraParams`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L104) — libfreenect2/include/libfreenect2/libfreenect2.hpp:104
    """

    fx: float
    """C++ source:

    - [`float fx; ///< Focal length x (pixel)`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L106) — libfreenect2/include/libfreenect2/libfreenect2.hpp:106
    """
    fy: float
    """C++ source:

    - [`float fy; ///< Focal length y (pixel)`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L107) — libfreenect2/include/libfreenect2/libfreenect2.hpp:107
    """
    cx: float
    """C++ source:

    - [`float cx; ///< Principal point x (pixel)`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L108) — libfreenect2/include/libfreenect2/libfreenect2.hpp:108
    """
    cy: float
    """C++ source:

    - [`float cy; ///< Principal point y (pixel)`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L109) — libfreenect2/include/libfreenect2/libfreenect2.hpp:109
    """
    k1: float
    """C++ source:

    - [`float k1; ///< Radial distortion coefficient, 1st-order`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L110) — libfreenect2/include/libfreenect2/libfreenect2.hpp:110
    """
    k2: float
    """C++ source:

    - [`float k2; ///< Radial distortion coefficient, 2nd-order`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L111) — libfreenect2/include/libfreenect2/libfreenect2.hpp:111
    """
    k3: float
    """C++ source:

    - [`float k3; ///< Radial distortion coefficient, 3rd-order`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L112) — libfreenect2/include/libfreenect2/libfreenect2.hpp:112
    """
    p1: float
    """C++ source:

    - [`float p1; ///< Tangential distortion coefficient`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L113) — libfreenect2/include/libfreenect2/libfreenect2.hpp:113
    """
    p2: float
    """C++ source:

    - [`float p2; ///< Tangential distortion coefficient`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L114) — libfreenect2/include/libfreenect2/libfreenect2.hpp:114
    """
    def __init__(self) -> None: ...

class Registration:
    """Python interface for ``libfreenect2::Registration``.

    C++ source:

    - [`class LIBFREENECT2_API Registration`](file:///home/coder/portland/libfreenect2/include/libfreenect2/registration.h#L54) — libfreenect2/include/libfreenect2/registration.h:54
    """

    def __init__(self, irparams: IrCameraParams, cparams: ColorCameraParams) -> None:
        """C++ source:

        - [`Registration::Registration(Freenect2Device::IrCameraParams depth_p, Freenect2Device::Co...`](file:///home/coder/portland/libfreenect2/src/registration.cpp#L362) — libfreenect2/src/registration.cpp:362
        """
    def apply(
        self,
        rgb: Frame,
        depth: Frame,
        undistorted: Frame,
        registered: Frame,
        enable_filter: bool = True,
        bigdepth: Frame | None = None,
        color_depth_map: npt.NDArray[np.int32] | None = None,
    ) -> None:
        """Map color onto depth.

        ``rgb`` 1920x1080, ``depth``/``undistorted``/``registered`` 512x424
        (bytes_per_pixel=4), ``bigdepth`` 1920x1082, ``color_depth_map``
        shape ``(424*512,)`` int32.

        C++ source:

        - [`void Registration::apply( int dx, int dy, float dz, float& cx, float &cy) const`](file:///home/coder/portland/libfreenect2/src/registration.cpp#L108) — libfreenect2/src/registration.cpp:108
        - [`void Registration::apply(const Frame *rgb, const Frame *depth, Frame *undistorted, Fram...`](file:///home/coder/portland/libfreenect2/src/registration.cpp#L123) — libfreenect2/src/registration.cpp:123
        """
    def undistortDepth(self, depth: Frame, undistorted: Frame) -> None:
        """C++ source:

        - [`void Registration::undistortDepth(const Frame *depth, Frame *undistorted) const`](file:///home/coder/portland/libfreenect2/src/registration.cpp#L276) — libfreenect2/src/registration.cpp:276
        """
    def getPointXYZRGB(
        self, undistorted: Frame, registered: Frame, r: int, c: int
    ) -> tuple[float, float, float, int, int, int]:
        """Returns ``(x, y, z, b, g, r)``; coordinates in meters.

        C++ source:

        - [`void Registration::getPointXYZRGB (const Frame* undistorted, const Frame* registered, i...`](file:///home/coder/portland/libfreenect2/src/registration.cpp#L317) — libfreenect2/src/registration.cpp:317
        """
    def getPointXYZ(self, undistorted: Frame, r: int, c: int) -> tuple[float, float, float]:
        """Returns ``(x, y, z)`` in meters.

        C++ source:

        - [`void Registration::getPointXYZ(const Frame *undistorted, int r, int c, float &x, float ...`](file:///home/coder/portland/libfreenect2/src/registration.cpp#L337) — libfreenect2/src/registration.cpp:337
        """

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class Logger:
    """Python interface for ``libfreenect2::Logger``.

    C++ source:

    - [`class LIBFREENECT2_API Logger`](file:///home/coder/portland/libfreenect2/include/libfreenect2/logger.h#L45) — libfreenect2/include/libfreenect2/logger.h:45
    """

    def level(self) -> int:
        """C++ source:

        - [`Logger::Level Logger::level() const`](file:///home/coder/portland/libfreenect2/src/logging.cpp#L80) — libfreenect2/src/logging.cpp:80
        """
    def log(self, level: LoggerLevel | int, message: bytes) -> None:
        """``message`` must be ``bytes`` (a ``str`` raises ``TypeError``).

        C++ source:

        - [`virtual void log(Level level, const std::string &message) = 0;`](file:///home/coder/portland/libfreenect2/include/libfreenect2/logger.h#L73) — libfreenect2/include/libfreenect2/logger.h:73
        """

def createConsoleLogger(level: LoggerLevel | int) -> Logger:
    """C++ source:

    - [`Logger *createConsoleLogger(Logger::Level level)`](file:///home/coder/portland/libfreenect2/src/logging.cpp#L125) — libfreenect2/src/logging.cpp:125
    """
def createConsoleLoggerWithDefaultLevel() -> Logger:
    """C++ source:

    - [`Logger *createConsoleLoggerWithDefaultLevel()`](file:///home/coder/portland/libfreenect2/src/logging.cpp#L130) — libfreenect2/src/logging.cpp:130
    """
def getGlobalLogger() -> Logger:
    """Always returns a ``Logger`` object, even after ``setGlobalLogger(None)``
    (it then wraps a NULL ``Logger*``; don't call methods on it).

    C++ source:

    - [`Logger *getGlobalLogger()`](file:///home/coder/portland/libfreenect2/src/logging.cpp#L180) — libfreenect2/src/logging.cpp:180
    """
def setGlobalLogger(logger: Logger | None = None) -> None:
    """Set the global logger; ``None`` disables logging.

    C++ source:

    - [`void setGlobalLogger(Logger *logger)`](file:///home/coder/portland/libfreenect2/src/logging.cpp#L185) — libfreenect2/src/logging.cpp:185
    """

# ---------------------------------------------------------------------------
# Packet pipelines
# ---------------------------------------------------------------------------

class PacketPipeline:
    """Base class for depth-processing pipelines.

    C++ source:

    - [`class LIBFREENECT2_API PacketPipeline`](file:///home/coder/portland/libfreenect2/include/libfreenect2/packet_pipeline.h#L54) — libfreenect2/include/libfreenect2/packet_pipeline.h:54
    """

class CpuPacketPipeline(PacketPipeline):
    """C++ source:

    - [`class LIBFREENECT2_API CpuPacketPipeline : public PacketPipeline`](file:///home/coder/portland/libfreenect2/include/libfreenect2/packet_pipeline.h#L86) — libfreenect2/include/libfreenect2/packet_pipeline.h:86
    """
    def __init__(self) -> None:
        """C++ source:

        - [`CpuPacketPipeline::CpuPacketPipeline()`](file:///home/coder/portland/libfreenect2/src/packet_pipeline.cpp#L131) — libfreenect2/src/packet_pipeline.cpp:131
        """

class OpenGLPacketPipeline(PacketPipeline):
    """C++ source:

    - [`class LIBFREENECT2_API OpenGLPacketPipeline : public PacketPipeline`](file:///home/coder/portland/libfreenect2/include/libfreenect2/packet_pipeline.h#L95) — libfreenect2/include/libfreenect2/packet_pipeline.h:95
    """
    def __init__(self, debug: bool = False) -> None:
        """C++ source:

        - [`OpenGLPacketPipeline::OpenGLPacketPipeline(void *parent_opengl_context, bool debug) : p...`](file:///home/coder/portland/libfreenect2/src/packet_pipeline.cpp#L139) — libfreenect2/src/packet_pipeline.cpp:139
        """

class CudaPacketPipeline(PacketPipeline):
    """C++ source:

    - [`class LIBFREENECT2_API CudaPacketPipeline : public PacketPipeline`](file:///home/coder/portland/libfreenect2/include/libfreenect2/packet_pipeline.h#L134) — libfreenect2/include/libfreenect2/packet_pipeline.h:134
    """
    def __init__(self, device_id: int = -1) -> None:
        """C++ source:

        - [`CudaPacketPipeline::CudaPacketPipeline(const int deviceId) : deviceId(deviceId)`](file:///home/coder/portland/libfreenect2/src/packet_pipeline.cpp#L166) — libfreenect2/src/packet_pipeline.cpp:166
        """

# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------

class Freenect2Device:
    """Python interface for ``libfreenect2::Freenect2Device``.

    Obtain via ``Freenect2.openDevice`` / ``Freenect2.openDefaultDevice``.

    C++ source:

    - [`class LIBFREENECT2_API Freenect2Device`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L48) — libfreenect2/include/libfreenect2/libfreenect2.hpp:48
    - [`class Freenect2DeviceImpl : public Freenect2Device`](file:///home/coder/portland/libfreenect2/src/libfreenect2.cpp#L209) — libfreenect2/src/libfreenect2.cpp:209
    """

    def getSerialNumber(self) -> bytes:
        """C++ source:

        - [`virtual std::string getSerialNumber() = 0;`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L132) — libfreenect2/include/libfreenect2/libfreenect2.hpp:132
        - [`std::string Freenect2DeviceImpl::getSerialNumber()`](file:///home/coder/portland/libfreenect2/src/libfreenect2.cpp#L692) — libfreenect2/src/libfreenect2.cpp:692
        """
    def getFirmwareVersion(self) -> bytes:
        """C++ source:

        - [`virtual std::string getFirmwareVersion() = 0;`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L133) — libfreenect2/include/libfreenect2/libfreenect2.hpp:133
        - [`std::string Freenect2DeviceImpl::getFirmwareVersion()`](file:///home/coder/portland/libfreenect2/src/libfreenect2.cpp#L697) — libfreenect2/src/libfreenect2.cpp:697
        """
    def getColorCameraParams(self) -> ColorCameraParams:
        """C++ source:

        - [`virtual ColorCameraParams getColorCameraParams() = 0;`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L138) — libfreenect2/include/libfreenect2/libfreenect2.hpp:138
        - [`Freenect2Device::ColorCameraParams Freenect2DeviceImpl::getColorCameraParams()`](file:///home/coder/portland/libfreenect2/src/libfreenect2.cpp#L702) — libfreenect2/src/libfreenect2.cpp:702
        """
    def getIrCameraParams(self) -> IrCameraParams:
        """C++ source:

        - [`virtual IrCameraParams getIrCameraParams() = 0;`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L143) — libfreenect2/include/libfreenect2/libfreenect2.hpp:143
        - [`Freenect2Device::IrCameraParams Freenect2DeviceImpl::getIrCameraParams()`](file:///home/coder/portland/libfreenect2/src/libfreenect2.cpp#L708) — libfreenect2/src/libfreenect2.cpp:708
        """
    def setColorFrameListener(self, listener: FrameListener) -> None:
        """C++ source:

        - [`virtual void setColorFrameListener(FrameListener* rgb_frame_listener) = 0;`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L160) — libfreenect2/include/libfreenect2/libfreenect2.hpp:160
        - [`void Freenect2DeviceImpl::setColorFrameListener(libfreenect2::FrameListener* rgb_frame_...`](file:///home/coder/portland/libfreenect2/src/libfreenect2.cpp#L743) — libfreenect2/src/libfreenect2.cpp:743
        """
    def setIrAndDepthFrameListener(self, listener: FrameListener) -> None:
        """C++ source:

        - [`virtual void setIrAndDepthFrameListener(FrameListener* ir_frame_listener) = 0;`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L163) — libfreenect2/include/libfreenect2/libfreenect2.hpp:163
        - [`void Freenect2DeviceImpl::setIrAndDepthFrameListener(libfreenect2::FrameListener* ir_fr...`](file:///home/coder/portland/libfreenect2/src/libfreenect2.cpp#L750) — libfreenect2/src/libfreenect2.cpp:750
        """
    def start(self) -> None:
        """Returns ``None``: the C++ ``bool`` result is discarded by the binding,
        so failure cannot be detected from Python.

        C++ source:

        - [`virtual bool start() = 0;`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L216) — libfreenect2/include/libfreenect2/libfreenect2.hpp:216
        - [`bool Freenect2DeviceImpl::start()`](file:///home/coder/portland/libfreenect2/src/libfreenect2.cpp#L886) — libfreenect2/src/libfreenect2.cpp:886
        """
    def startStreams(self, rgb: bool, depth: bool) -> None:
        """Returns ``None``: the C++ ``bool`` result is discarded by the binding,
        so failure cannot be detected from Python.

        C++ source:

        - [`virtual bool startStreams(bool rgb, bool depth) = 0;`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L225) — libfreenect2/include/libfreenect2/libfreenect2.hpp:225
        - [`bool Freenect2DeviceImpl::startStreams(bool enable_rgb, bool enable_depth)`](file:///home/coder/portland/libfreenect2/src/libfreenect2.cpp#L891) — libfreenect2/src/libfreenect2.cpp:891
        """
    def stop(self) -> None:
        """Returns ``None``: the C++ ``bool`` result is discarded by the binding.

        C++ source:

        - [`virtual bool stop() = 0;`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L231) — libfreenect2/include/libfreenect2/libfreenect2.hpp:231
        - [`bool Freenect2DeviceImpl::stop()`](file:///home/coder/portland/libfreenect2/src/libfreenect2.cpp#L987) — libfreenect2/src/libfreenect2.cpp:987
        """
    def close(self) -> None:
        """Returns ``None``: the C++ ``bool`` result is discarded by the binding.

        C++ source:

        - [`virtual bool close() = 0;`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L237) — libfreenect2/include/libfreenect2/libfreenect2.hpp:237
        - [`bool Freenect2DeviceImpl::close()`](file:///home/coder/portland/libfreenect2/src/libfreenect2.cpp#L1030) — libfreenect2/src/libfreenect2.cpp:1030
        """

class Freenect2:
    """Python interface for ``libfreenect2::Freenect2``.

    C++ source:

    - [`class LIBFREENECT2_API Freenect2`](file:///home/coder/portland/libfreenect2/include/libfreenect2/libfreenect2.hpp#L254) — libfreenect2/include/libfreenect2/libfreenect2.hpp:254
    """

    def __init__(self) -> None:
        """C++ source:

        - [`Freenect2::Freenect2(void *usb_context) :`](file:///home/coder/portland/libfreenect2/src/libfreenect2.cpp#L1133) — libfreenect2/src/libfreenect2.cpp:1133
        """
    def enumerateDevices(self) -> int:
        """C++ source:

        - [`int Freenect2::enumerateDevices()`](file:///home/coder/portland/libfreenect2/src/libfreenect2.cpp#L1143) — libfreenect2/src/libfreenect2.cpp:1143
        """
    def getDeviceSerialNumber(self, idx: int) -> bytes:
        """C++ source:

        - [`std::string Freenect2::getDeviceSerialNumber(int idx)`](file:///home/coder/portland/libfreenect2/src/libfreenect2.cpp#L1149) — libfreenect2/src/libfreenect2.cpp:1149
        """
    def getDefaultDeviceSerialNumber(self) -> bytes:
        """C++ source:

        - [`std::string Freenect2::getDefaultDeviceSerialNumber()`](file:///home/coder/portland/libfreenect2/src/libfreenect2.cpp#L1159) — libfreenect2/src/libfreenect2.cpp:1159
        """
    def openDevice(
        self, name: int | bytes, pipeline: PacketPipeline | None = None
    ) -> Freenect2Device:
        """Open device by serial number (``bytes``) or index (``int``).

        A ``str`` serial raises ``TypeError``; use e.g.
        ``getDefaultDeviceSerialNumber()`` (already ``bytes``) or ``s.encode()``.
        An unknown serial does not raise: it returns a ``Freenect2Device``
        wrapping a NULL pointer, so check ``enumerateDevices()`` first.

        C++ source:

        - [`Freenect2Device *Freenect2::openDevice(int idx)`](file:///home/coder/portland/libfreenect2/src/libfreenect2.cpp#L1164) — libfreenect2/src/libfreenect2.cpp:1164
        - [`Freenect2Device *Freenect2::openDevice(int idx, const PacketPipeline *pipeline)`](file:///home/coder/portland/libfreenect2/src/libfreenect2.cpp#L1169) — libfreenect2/src/libfreenect2.cpp:1169
        - [`Freenect2Device *Freenect2::openDevice(const std::string &serial)`](file:///home/coder/portland/libfreenect2/src/libfreenect2.cpp#L1273) — libfreenect2/src/libfreenect2.cpp:1273
        - [`Freenect2Device *Freenect2::openDevice(const std::string &serial, const PacketPipeline ...`](file:///home/coder/portland/libfreenect2/src/libfreenect2.cpp#L1278) — libfreenect2/src/libfreenect2.cpp:1278
        """
    def openDefaultDevice(self, pipeline: PacketPipeline | None = None) -> Freenect2Device:
        """C++ source:

        - [`Freenect2Device *Freenect2::openDefaultDevice()`](file:///home/coder/portland/libfreenect2/src/libfreenect2.cpp#L1295) — libfreenect2/src/libfreenect2.cpp:1295
        - [`Freenect2Device *Freenect2::openDefaultDevice(const PacketPipeline *pipeline)`](file:///home/coder/portland/libfreenect2/src/libfreenect2.cpp#L1300) — libfreenect2/src/libfreenect2.cpp:1300
        """
