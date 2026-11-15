"""Type stubs for pylibfreenect2 0.1.4 (hand-written from the module's docstrings
and runtime introspection; the real implementation is a Cython extension)."""

from enum import IntEnum

from .libfreenect2 import *

__version__: str

class FrameType(IntEnum):
    """Python-side enum for ``libfreenect2::Frame::Type``.

    C++ source:

    - [`enum Type`](file:///home/coder/portland/libfreenect2/include/libfreenect2/frame_listener.hpp#L48) — libfreenect2/include/libfreenect2/frame_listener.hpp:48
    """

    Color = 1
    """C++ source:

    - [`Color = 1, ///< 1920x1080. BGRX or RGBX.`](file:///home/coder/portland/libfreenect2/include/libfreenect2/frame_listener.hpp#L50) — libfreenect2/include/libfreenect2/frame_listener.hpp:50
    """
    Ir = 2
    """C++ source:

    - [`Ir = 2,    ///< 512x424 float. Range is [0.0, 65535.0].`](file:///home/coder/portland/libfreenect2/include/libfreenect2/frame_listener.hpp#L51) — libfreenect2/include/libfreenect2/frame_listener.hpp:51
    """
    Depth = 4
    """C++ source:

    - [`Depth = 4  ///< 512x424 float, unit: millimeter. Non-positive, NaN, and infinity are in...`](file:///home/coder/portland/libfreenect2/include/libfreenect2/frame_listener.hpp#L52) — libfreenect2/include/libfreenect2/frame_listener.hpp:52
    """

class LoggerLevel(IntEnum):
    """Python-side enum for ``libfreenect2::Logger::Level``.

    C++ source:

    - [`enum Level`](file:///home/coder/portland/libfreenect2/include/libfreenect2/logger.h#L49) — libfreenect2/include/libfreenect2/logger.h:49
    """

    NONE = 0
    """C++ source:

    - [`None = 0,`](file:///home/coder/portland/libfreenect2/include/libfreenect2/logger.h#L51) — libfreenect2/include/libfreenect2/logger.h:51
    """
    Error = 1
    """C++ source:

    - [`Error = 1,`](file:///home/coder/portland/libfreenect2/include/libfreenect2/logger.h#L52) — libfreenect2/include/libfreenect2/logger.h:52
    """
    Warning = 2
    """C++ source:

    - [`Warning = 2,`](file:///home/coder/portland/libfreenect2/include/libfreenect2/logger.h#L53) — libfreenect2/include/libfreenect2/logger.h:53
    """
    Info = 3
    """C++ source:

    - [`Info = 3,`](file:///home/coder/portland/libfreenect2/include/libfreenect2/logger.h#L54) — libfreenect2/include/libfreenect2/logger.h:54
    """
    Debug = 4
    """C++ source:

    - [`Debug = 4,`](file:///home/coder/portland/libfreenect2/include/libfreenect2/logger.h#L55) — libfreenect2/include/libfreenect2/logger.h:55
    """
