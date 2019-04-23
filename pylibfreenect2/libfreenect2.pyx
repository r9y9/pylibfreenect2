# coding: utf-8
# cython: boundscheck=True, wraparound=True

"""
API
===

.. important::
    Python API's are designed to minimize differences in C++ and Python; i.e. all
    classes and methods should have the same name; function signatures should
    also be same as possible. For the slight differences, see below in details.


All functionality in ``pylibfreenect2.libfreenect2`` is directly accesible from
the top-level ``pylibfreenect2.*`` namespace.

The sections below are organized by following
`the offical docs <https://openkinect.github.io/libfreenect2/modules.html>`_.


Frame Listeners
---------------

FrameType
^^^^^^^^^

.. autoclass:: pylibfreenect2.FrameType
    :members:

Frame
^^^^^

.. autoclass:: Frame
    :members:

FrameMap
^^^^^^^^

.. autoclass:: FrameMap
    :members:
    :special-members: __getitem__

SyncMultiFrameListener
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: SyncMultiFrameListener
    :members:

Initialization and Device Control
---------------------------------

Freenect2Device
^^^^^^^^^^^^^^^

.. autoclass:: Freenect2Device
    :members:

Freenect2
^^^^^^^^^

.. autoclass:: Freenect2
    :members:

ColorCameraParams
^^^^^^^^^^^^^^^^^

.. autoclass:: ColorCameraParams
    :members:

IrCameraParams
^^^^^^^^^^^^^^

.. autoclass:: IrCameraParams
    :members:

Logging utilities
-----------------

LoggerLevel
^^^^^^^^^^^

.. autoclass:: pylibfreenect2.LoggerLevel
    :members:

Logger
^^^^^^

.. autoclass:: Logger
    :members:

Functions
^^^^^^^^^

.. autosummary::
    :toctree: generated/

    createConsoleLogger
    createConsoleLoggerWithDefaultLevel
    getGlobalLogger
    setGlobalLogger

Packet Pipelines
----------------

PacketPipeline
^^^^^^^^^^^^^^

.. autoclass:: PacketPipeline
    :members:

CpuPacketPipeline
^^^^^^^^^^^^^^^^^

.. autoclass:: CpuPacketPipeline
    :members:

OpenCLPacketPipeline
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: OpenCLPacketPipeline
    :members:

OpenCLKdePacketPipeline
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: OpenCLKdePacketPipeline
    :members:

OpenGLPacketPipeline
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: OpenGLPacketPipeline
    :members:

Registration and Geometry
-------------------------

Registration
^^^^^^^^^^^^

.. autoclass:: Registration
    :members:
"""


import numpy as np

cimport numpy as np
np.import_array()

cimport cython

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.map cimport map

from libcpp.cast cimport reinterpret_cast
from libc.stdint cimport uint8_t

# Workaround for use of pointer type in reinterpret_cast
# https://groups.google.com/forum/#!msg/cython-users/FgEf7Vrx4AM/dm7WY_bMCAAJ
ctypedef uint8_t* uint8_pt

# Import libfreenect2 definitions
from libfreenect2 cimport libfreenect2

# A workaround to access nested cppclass that externed in a separate namespace.
# Nested cppclass Freenect2Device::ColorCameraParams cannot be accesed
# with chained ‘.‘ access (i.e.
# `libfreenect2.Freenect2Device.ColorCameraParams`), here I explicitly import
#  Freenect2Device as _Freenect2Device (to avoid name conflict) and use
# `_Freenect2Device.ColorCameraParams` to access nested cppclass
# ColorCameraParams.
from libfreenect2.libfreenect2 cimport Freenect2Device as _Freenect2Device

from pylibfreenect2 import FrameType

cdef class Frame:
    """Python interface for ``libfreenect2::Frame``.

    The Frame is a container of the C++ pointer ``libfreenect2::Frame*``.

    .. note::
        By default, Frame just keeps a pointer of ``libfreenect2::Frame`` that
        should be allocated and released by SyncMultiFrameListener (i.e. Frame
        itself doesn't own the allocated memory) as in C++. However, if Frame is
        created by providing ``width``, ``height`` and ``bytes_per_pixel``, then
        it allocates necessary memory in ``__cinit__`` and release it in
        ``__dealloc__`` method.

    Attributes
    ----------
    ptr : libfreenect2::Frame*
        Pointer of Frame.

    take_ownership : bool
        If True, the class instance allocates memory for Frame* and release it
        in ``__dealloc__``. If `width`, `height` and `bytes_per_pixel` are given
        in ``__cinit__``, which is necessary to allocate how much memory we need,
        ``take_ownership`` is set to True internally, otherwise False. Note that
        the value itself cannot be set by users.

    frame_type : int
        Underlying frame type.

    Parameters
    ----------
    width : int, optional
        Width of Frame. Default is None.

    height : int, optional
        Height of Frame. Default is None.

    bytes_per_pixel : int, optional
        Bytes per pixels of Frame. Default is None.

    frame_type : int, optional
        Underlying frame type. Default is -1. Used by ``asarray`` method.

    numpy_array : numpy.ndarray, optional
        Numpy array of depth or ir data with ndim=2,
        that will be converted to a frame class.
        Default is None.

    See also
    --------

    pylibfreenect2.FrameType
    """

    cdef libfreenect2.Frame* ptr
    cdef bool take_ownership
    cdef int frame_type

    def __cinit__(self, width=None, height=None, bytes_per_pixel=None,
            int frame_type=-1, np.ndarray[np.float32_t, ndim=2, mode="c"] numpy_array=None):
        w,h,b = width, height, bytes_per_pixel
        all_none = (w is None) and (h is None) and (b is None)
        all_not_none = (w is not None) and (h is not None) and (b is not None)
        assert all_none or all_not_none

        self.frame_type = frame_type

        if all_not_none:
            self.take_ownership = True
            if numpy_array is None:
                self.ptr = new libfreenect2.Frame(
                    width, height, bytes_per_pixel, NULL)
            else:
                self.__instantiate_frame_with_bytes(
                    width, height, bytes_per_pixel, numpy_array.reshape(-1))
        else:
            self.take_ownership = False

    cdef __instantiate_frame_with_bytes(self, int width, int height,
        int bytes_per_pixel, np.ndarray[np.float32_t, ndim=1, mode="c"] numpy_array):
        cdef uint8_t* bytes_ptr = reinterpret_cast[uint8_pt](&numpy_array[0])
        self.ptr = new libfreenect2.Frame(
            width, height, bytes_per_pixel, bytes_ptr)

    def __dealloc__(self):
        if self.take_ownership and self.ptr is not NULL:
            del self.ptr

    @property
    def timestamp(self):
        """Same as ``libfreenect2::Frame::timestamp``"""
        return self.ptr.timestamp

    @property
    def sequence(self):
        """Same as ``libfreenect2::Frame::sequence``"""
        return self.ptr.sequence

    @property
    def width(self):
        """Same as ``libfreenect2::Frame::width``"""
        return self.ptr.width

    @property
    def height(self):
        """Same as ``libfreenect2::Frame::height``"""
        return self.ptr.height

    @property
    def bytes_per_pixel(self):
        """Same as ``libfreenect2::Frame::bytes_per_pixel``"""
        return self.ptr.bytes_per_pixel

    @property
    def exposure(self):
        """Same as ``libfreenect2::Frame::exposure``"""
        return self.ptr.exposure

    @property
    def gain(self):
        """Same as ``libfreenect2::Frame::gain``"""
        return self.ptr.gain

    @property
    def gamma(self):
        """Same as ``libfreenect2::Frame::gamma``"""
        return self.ptr.gamma

    cdef __uint8_data(self):
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.ptr.height
        shape[1] = <np.npy_intp> self.ptr.width
        shape[2] = <np.npy_intp> 4
        cdef np.ndarray array = np.PyArray_SimpleNewFromData(
            3, shape, np.NPY_UINT8, self.ptr.data)

        return array

    cdef __float32_data(self):
        cdef np.npy_intp shape[2]
        shape[0] = <np.npy_intp> self.ptr.height
        shape[1] = <np.npy_intp> self.ptr.width
        cdef np.ndarray array = np.PyArray_SimpleNewFromData(
            2, shape, np.NPY_FLOAT32, self.ptr.data)

        return array

    def __asarray(self, dtype):
        if dtype != np.uint8 and dtype != np.float32:
            raise ValueError("np.uint8 or np.float32 is only supported")
        if dtype == np.uint8:
            return self.__uint8_data()
        else:
            return self.__float32_data()

    def asarray(self, dtype=None):
        """Frame to ``numpy.ndarray`` conversion

        Internal data of Frame can be represented as:

        - 3d array of ``numpy.uint8`` for color
        - 2d array of ``numpy.float32`` for IR and depth

        Parameters
        ----------
        dtype : numpy dtype, optional
             Data type (``numpy.uint8`` or ``numpy.float32``). If None, data
             type is automatically selected if possible. Default is None.

        Returns
        -------
        array : ``numpy.ndarray``, shape: ``(height, width)`` for IR and depth,
        ``(4, height, width)`` for Color.
            Array of internal frame.

        Raises
        ------
        ValueError
            - If dtype is None and underlying frame type cannot be determined.
            - If dtype neither ``numpy.uint8`` nor ``numpy.float32`` is specified

        Examples
        --------

        .. code-block:: python

            rgb_array = frames["color"].asarray()
            ir_array = frames["ir"].asarray()
            depth_array = frames["depth"].asarray()

        .. code-block:: python

            undistorted = Frame(512, 424, 4)
            registered = Frame(512, 424, 4)
            undistorted_arrray = undistorted.asarray(dtype=np.float32)
            registered_array = registered.asarray(dtype=np.uint8)

        """
        if dtype is None and self.frame_type < 0:
            raise ValueError("Cannot determine type of data. Specify dtype explicitly.")

        if dtype is None:
            if self.frame_type == FrameType.Color:
                return self.__asarray(np.uint8)
            elif self.frame_type == FrameType.Ir or self.frame_type == FrameType.Depth:
                return self.__asarray(np.float32)
            else:
                assert False
        else:
            return self.__asarray(dtype)

cdef class FrameListener:
    cdef libfreenect2.FrameListener* listener_ptr_alias


cdef intenum_to_frame_type(int n):
    if n == FrameType.Color:
        return libfreenect2.Color
    elif n == FrameType.Ir:
        return libfreenect2.Ir
    elif n == FrameType.Depth:
        return libfreenect2.Depth
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
    """Python interface for ``libfreenect2::FrameMap``.

    The FrameMap is a container of C++ value ``libfreenect2::FrameMap`` (aliased
    to ``std::map<libfreenect2::Frame::Type,libfreenect2::Frame*>`` in C++).

    .. note::
        By default, FrameMap just keeps a reference of ``libfreenect2::FrameMap``
        that should be allcoated and released by SyncMultiFrameListener (i.e.
        FrameMap itself doesn't own the allocated memory) as in C++.

    Attributes
    ----------
    internal_frame_map : std::map<libfreenect2::Frame::Type, libfreenect2::Frame*>
        Internal FrameMap.

    """
    cdef map[libfreenect2.LibFreenect2FrameType, libfreenect2.Frame*] internal_frame_map
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
        """Get access to the internal FrameMap.

        This allows the following dict-like syntax:

        .. code-block:: python

            color = frames[pylibfreenect2.FrameType.Color]

        .. code-block:: python

            color = frames['color']

        .. code-block:: python

            color = frames[1] # with IntEnum value

        The key can be of ``FrameType`` (a subclass of IntEnum), str or int type
        as shown above.

        Parameters
        ----------
        key : ``FrameType``, str or int
            Key for the internal FrameMap. available str keys are ``color``,
            ``ir`` and ``depth``.

        Returns
        -------
        frame : Frame
            Frame for the specified key.

        Raises
        ------
        KeyError
            if unknown key is specified

        See also
        --------

        pylibfreenect2.FrameType

        """
        cdef libfreenect2.LibFreenect2FrameType frame_type
        cdef intkey

        if isinstance(key, int) or isinstance(key, FrameType):
            frame_type = intenum_to_frame_type(key)
            intkey = key
        elif isinstance(key, str):
            frame_type = str_to_frame_type(key)
            intkey = str_to_int_frame_type(key)
        else:
            raise KeyError("")

        cdef libfreenect2.Frame* frame_ptr = self.internal_frame_map[frame_type]
        cdef Frame frame = Frame(frame_type=intkey)
        frame.ptr = frame_ptr
        return frame


cdef class SyncMultiFrameListener(FrameListener):
    """Python interface for ``libfreenect2::SyncMultiFrameListener``.

    The SyncMultiFrameListener is a container of
    C++ pointer ``libfreenect2::SyncMultiFrameListener*``. The pointer of
    SyncMultiFrameListener is allocated in ``__cinit__`` and released in
    ``__dealloc__`` method.

    Parameters
    ----------
    frame_types : unsigned int, optional
        Frame types that we want to listen. It can be logical OR of:

            - ``FrameType.Color``
            - ``FrameType.Ir``
            - ``FrameType.Depth``

        Default is ``FrameType.Color | FrameType.Ir | FrameType.Depth``

    Attributes
    ----------
    ptr : libfreenect2.SyncMultiFrameListener*
        Pointer of ``libfreenect2::SyncMultiFrameListener``

    listener_ptr_alias : libfreenect2.FrameListener*
        Pointer of ``libfreenect2::FrameListener``. This is necessary to call
        methods that operate on ``libfreenect2::FrameListener*``, not
        ``libfreenect2::SyncMultiFrameListener``.

    See also
    --------

    pylibfreenect2.FrameType

    """

    cdef libfreenect2.SyncMultiFrameListener* ptr

    def __cinit__(self, unsigned int frame_types=<unsigned int>(
                        FrameType.Color | FrameType.Ir | FrameType.Depth)):
        self.ptr = new libfreenect2.SyncMultiFrameListener(frame_types)
        self.listener_ptr_alias = <libfreenect2.FrameListener*> self.ptr

    def __dealloc__(self):
        if self.ptr is not NULL:
            del self.ptr

    def hasNewFrame(self):
        """Same as ``libfreenect2::SyncMultiFrameListener::hasNewFrame()``.

        Returns
        -------
        r : Bool
            True if SyncMultiFrameListener has a new frame, False otherwise.
        """
        return self.ptr.hasNewFrame()

    def waitForNewFrame(self, FrameMap frame_map=None):
        """Same as ``libfreenect2::SyncMultiFrameListener::waitForNewFrame(Frame&)``.

        .. warning::

            Function signature can be different between Python and C++.

        Parameters
        ----------
        frame_map : FrameMap, optional
            If not None, SyncMultiFrameListener write to it inplace, otherwise
            a new FrameMap is allocated within the function and then returned.

        Returns
        -------
        frame_map : FrameMap
            FrameMap.

            .. note::
                FrameMap must be releaseed by call-side by calling ``release``
                function.

        Examples
        --------

        Suppose the following C++ code:

        .. code-block:: c++

            libfreenect2::FrameMap frames;
            listener->waitForNewFrame(frames);

        This can be translated in Python as follows:

        .. code-block:: python

            frames = listener.waitForNewFrame()

        or you can write it more similar to C++:

        .. code-block:: python

            frames = pylibfreenect2.FrameMap()
            listener.waitForNewFrame(frames)

        """
        if frame_map is None:
            frame_map = FrameMap(take_ownership=False)
        with nogil:
            self.ptr.waitForNewFrame(frame_map.internal_frame_map)
        return frame_map


    def release(self, FrameMap frame_map):
        """Same as ``libfreenect2::SyncMultiFrameListener::release(Frame&)``.

        Parameters
        ----------
        frame_map : FrameMap
            FrameMap.
        """
        self.ptr.release(frame_map.internal_frame_map)


cdef class ColorCameraParams:
    """Python interface for ``libfreenect2::Freenect2Device::ColorCameraParams``.

    Attributes
    ----------
    params : ``libfreenect2::Freenect2Device::ColorCameraParams``

    See also
    --------
    pylibfreenect2.libfreenect2.Freenect2Device.getColorCameraParams
    """
    cdef _Freenect2Device.ColorCameraParams params

    # TODO: wrap all instance variables
    @property
    def fx(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::fx``"""
        return self.params.fx

    @fx.setter
    def fx(self, value):
        """Sets fx parameter"""
        self.params.fx = value

    @property
    def fy(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::fy``"""
        return self.params.fy

    @fy.setter
    def fy(self, value):
        """Sets fy parameter"""
        self.params.fy = value

    @property
    def cx(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::cx``"""
        return self.params.cx

    @cx.setter
    def cx(self, value):
        """Sets cx parameter"""
        self.params.cx = value

    @property
    def cy(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::cy``"""
        return self.params.cy

    @cy.setter
    def cy(self, value):
        """Sets cx parameter"""
        self.params.cy = value

    @property
    def shift_d(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::shift_d``"""
        return self.params.shift_d

    @shift_d.setter
    def shift_d(self, value):
        """Sets shift_d parameter"""
        self.params.shift_d = value


    @property
    def shift_m(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::shift_m``"""
        return self.params.shift_m

    @shift_m.setter
    def shift_m(self, value):
        """Sets shift_m parameter"""
        self.params.shift_m = value


    @property
    def mx_x3y0(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::mx_x3y0``"""
        return self.params.mx_x3y0

    @mx_x3y0.setter
    def mx_x3y0(self, value):
        """Sets mx_x3y0 parameter"""
        self.params.mx_x3y0 = value


    @property
    def mx_x0y3(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::mx_x0y3``"""
        return self.params.mx_x0y3

    @mx_x0y3.setter
    def mx_x0y3(self, value):
        """Sets mx_x0y3 parameter"""
        self.params.mx_x0y3 = value


    @property
    def mx_x2y1(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::mx_x2y1``"""
        return self.params.mx_x2y1

    @mx_x2y1.setter
    def mx_x2y1(self, value):
        """Sets mx_x2y1 parameter"""
        self.params.mx_x2y1 = value


    @property
    def mx_x1y2(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::mx_x1y2``"""
        return self.params.mx_x1y2

    @mx_x1y2.setter
    def mx_x1y2(self, value):
        """Sets mx_x1y2 parameter"""
        self.params.mx_x1y2 = value


    @property
    def mx_x2y0(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::mx_x2y0``"""
        return self.params.mx_x2y0

    @mx_x2y0.setter
    def mx_x2y0(self, value):
        """Sets mx_x2y0 parameter"""
        self.params.mx_x2y0 = value


    @property
    def mx_x0y2(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::mx_x0y2``"""
        return self.params.mx_x0y2

    @mx_x0y2.setter
    def mx_x0y2(self, value):
        """Sets mx_x0y2 parameter"""
        self.params.mx_x0y2 = value


    @property
    def mx_x1y1(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::mx_x1y1``"""
        return self.params.mx_x1y1

    @mx_x1y1.setter
    def mx_x1y1(self, value):
        """Sets mx_x1y1 parameter"""
        self.params.mx_x1y1 = value


    @property
    def mx_x1y0(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::mx_x1y0``"""
        return self.params.mx_x1y0

    @mx_x1y0.setter
    def mx_x1y0(self, value):
        """Sets mx_x1y0 parameter"""
        self.params.mx_x1y0 = value


    @property
    def mx_x0y1(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::mx_x0y1``"""
        return self.params.mx_x0y1

    @mx_x0y1.setter
    def mx_x0y1(self, value):
        """Sets mx_x0y1 parameter"""
        self.params.mx_x0y1 = value


    @property
    def mx_x0y0(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::mx_x0y0``"""
        return self.params.mx_x0y0

    @mx_x0y0.setter
    def mx_x0y0(self, value):
        """Sets mx_x0y0 parameter"""
        self.params.mx_x0y0 = value


    @property
    def my_x3y0(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::my_x3y0``"""
        return self.params.my_x3y0

    @my_x3y0.setter
    def my_x3y0(self, value):
        """Sets my_x3y0 parameter"""
        self.params.my_x3y0 = value


    @property
    def my_x0y3(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::my_x0y3``"""
        return self.params.my_x0y3

    @my_x0y3.setter
    def my_x0y3(self, value):
        """Sets my_x0y3 parameter"""
        self.params.my_x0y3 = value


    @property
    def my_x2y1(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::my_x2y1``"""
        return self.params.my_x2y1

    @my_x2y1.setter
    def my_x2y1(self, value):
        """Sets my_x2y1 parameter"""
        self.params.my_x2y1 = value


    @property
    def my_x1y2(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::my_x1y2``"""
        return self.params.my_x1y2

    @my_x1y2.setter
    def my_x1y2(self, value):
        """Sets my_x1y2 parameter"""
        self.params.my_x1y2 = value


    @property
    def my_x2y0(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::my_x2y0``"""
        return self.params.my_x2y0

    @my_x2y0.setter
    def my_x2y0(self, value):
        """Sets my_x2y0 parameter"""
        self.params.my_x2y0 = value


    @property
    def my_x0y2(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::my_x0y2``"""
        return self.params.my_x0y2

    @my_x0y2.setter
    def my_x0y2(self, value):
        """Sets my_x0y2 parameter"""
        self.params.my_x0y2 = value


    @property
    def my_x1y1(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::my_x1y1``"""
        return self.params.my_x1y1

    @my_x1y1.setter
    def my_x1y1(self, value):
        """Sets my_x1y1 parameter"""
        self.params.my_x1y1 = value


    @property
    def my_x1y0(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::my_x1y0``"""
        return self.params.my_x1y0

    @my_x1y0.setter
    def my_x1y0(self, value):
        """Sets my_x1y0 parameter"""
        self.params.my_x1y0 = value


    @property
    def my_x0y1(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::my_x0y1``"""
        return self.params.my_x0y1

    @my_x0y1.setter
    def my_x0y1(self, value):
        """Sets my_x0y1 parameter"""
        self.params.my_x0y1 = value


    @property
    def my_x0y0(self):
        """Same as ``libfreenect2::Freenect2Device::ColorCameraParams::my_x0y0``"""
        return self.params.my_x0y0

    @my_x0y0.setter
    def my_x0y0(self, value):
        """Sets my_x0y0 parameter"""
        self.params.my_x0y0 = value



cdef class IrCameraParams:
    """Python interface for ``libfreenect2::IrCameraParams``.

    Attributes
    ----------
    params : ``libfreenect2::Freenect2Device::IrCameraParams``

    See also
    --------
    pylibfreenect2.libfreenect2.Freenect2Device.getIrCameraParams
    """
    cdef _Freenect2Device.IrCameraParams params

    @property
    def fx(self):
        """Same as ``libfreenect2::Freenect2Device::IrCameraParams::fx``"""
        return self.params.fx

    @fx.setter
    def fx(self, value):
        """Sets fx parameter"""
        self.params.fx = value

    @property
    def fy(self):
        """Same as ``libfreenect2::Freenect2Device::IrCameraParams::fy``"""
        return self.params.fy

    @fy.setter
    def fy(self, value):
        """Sets fy parameter"""
        self.params.fy = value

    @property
    def cx(self):
        """Same as ``libfreenect2::Freenect2Device::IrCameraParams::cx``"""
        return self.params.cx

    @cx.setter
    def cx(self, value):
        """Sets cx parameter"""
        self.params.cx = value

    @property
    def cy(self):
        """Same as ``libfreenect2::Freenect2Device::IrCameraParams::cy``"""
        return self.params.cy

    @cy.setter
    def cy(self, value):
        """Sets cx parameter"""
        self.params.cy = value

    @property
    def k1(self):
        """Same as ``libfreenect2::Freenect2Device::IrCameraParams::k1``"""
        return self.params.k1

    @k1.setter
    def k1(self, value):
        """Sets k1 parameter"""
        self.params.k1 = value

    @property
    def k2(self):
        """Same as ``libfreenect2::Freenect2Device::IrCameraParams::k2``"""
        return self.params.k2

    @k2.setter
    def k2(self, value):
        """Sets k2 parameter"""
        self.params.k2 = value

    @property
    def k3(self):
        """Same as ``libfreenect2::Freenect2Device::IrCameraParams::k3``"""
        return self.params.k3

    @k3.setter
    def k3(self, value):
        """Sets k3 parameter"""
        self.params.k3 = value

    @property
    def p1(self):
        """Same as ``libfreenect2::Freenect2Device::IrCameraParams::p1``"""
        return self.params.p1

    @p1.setter
    def p1(self, value):
        """Sets p1 parameter"""
        self.params.p1 = value

    @property
    def p2(self):
        """Same as ``libfreenect2::Freenect2Device::IrCameraParams::p2``"""
        return self.params.p2

    @p2.setter
    def p2(self, value):
        """Sets p2 parameter"""
        self.params.p2 = value

cdef class Registration:
    """Python interface for ``libfreenect2::Registration``.

    The Registration is a container of C++ pointer
    ``libfreenect2::Registration*``. The pointer of Registration is allocated
    in ``__cinit__`` and released in ``__dealloc__`` method.

    Attributes
    ----------
    ptr : ``libfreenect2::Registration*``

    Parameters
    ----------
    irparams : IrCameraParams
        IR camera parameters.

    cparams : ColorCameraParams
        Color camera parameters.

    See also
    --------
    pylibfreenect2.libfreenect2.IrCameraParams
    pylibfreenect2.libfreenect2.ColorCameraParams
    pylibfreenect2.libfreenect2.Freenect2Device.getIrCameraParams
    pylibfreenect2.libfreenect2.Freenect2Device.getColorCameraParams
    """
    cdef libfreenect2.Registration* ptr

    def __cinit__(self, IrCameraParams irparams, ColorCameraParams cparams):
        cdef _Freenect2Device.IrCameraParams i = irparams.params
        cdef _Freenect2Device.ColorCameraParams c = cparams.params
        self.ptr = new libfreenect2.Registration(i, c)

    def __dealloc__(self):
        if self.ptr is not NULL:
            del self.ptr

    def apply(self, Frame rgb, Frame depth, Frame undistorted,
            Frame registered, enable_filter=True, Frame bigdepth=None,
            np.ndarray[np.int32_t, ndim=1, mode="c"] color_depth_map=None):
        """Same as ``libfreenect2::Registration::apply``.

        Parameters
        ----------
        rgb : Frame
            ``(1920, 1080)`` color frame

        depth : Frame
            ``(512, 424)`` depth frame

        undistorted : Frame
            ``(512, 424)`` registered depth frame

        registered : Frame
            ``(512, 424)`` registered color frame

        enable_filter : Bool, optional

        bigdepth : Frame, optional
            ``(1920, 1082)`` bigdepth frame

        color_depth_map : ``numpy.ndarray``, optional
            Array of shape: ``(424*512,)``, dtype ``np.int32``

        Raises
        ------
        ValueError
            If invalid shape of frame/array is provided

        """
        if rgb.width != 1920 or rgb.height != 1080:
            raise ValueError(
                "Shape of color frame {}x{} is invalid. Expected 1920x1080.".format(
                rgb.width, rgb.height))
        if depth.width != 512 or depth.height != 424:
            raise ValueError(
                "Shape of depth frame {}x{} is invalid. Expected 512x424.".format(
                depth.width, depth.height))
        if undistorted.width != 512 or undistorted.height != 424:
            raise ValueError(
                "Shape of undistorted frame {}x{} is invalid. Expected 512x424.".format(
                undistorted.width, undistorted.height))
        if registered.width != 512 or registered.height != 424:
            raise ValueError(
                "Shape of registered frame {}x{} is invalid. Expected 512x424.".format(
                registered.width, registered.height))

        if bigdepth is not None:
            if bigdepth.width != 1920 or bigdepth.height != 1082:
                raise ValueError(
                    "Shape of bigdepth frame {}x{} is invalid. Expected 1920x1082.".format(
                    bigdepth.width, bigdepth.height))

        if color_depth_map is not None:
            if color_depth_map.shape[0] != 424*512:
                raise ValueError(
                    "Shape of color_depth_map array ({},) is invalid. Expected (424*512,)".format(
                    color_depth_map.shape[0]))

        assert rgb.take_ownership == False
        assert depth.take_ownership == False
        assert undistorted.take_ownership == True
        assert registered.take_ownership == True
        assert bigdepth is None or bigdepth.take_ownership == True

        cdef libfreenect2.Frame* bigdepth_ptr = <libfreenect2.Frame*>(NULL) \
            if bigdepth is None else bigdepth.ptr
        cdef int* color_depth_map_ptr = <int*>(NULL) if color_depth_map is None \
            else <int*>(&color_depth_map[0])

        self.ptr.apply(rgb.ptr, depth.ptr, undistorted.ptr, registered.ptr,
            enable_filter, bigdepth_ptr, color_depth_map_ptr)

    def undistortDepth(self, Frame depth, Frame undistorted):
        """Same as ``libfreenect2::Registration::undistortDepth(bool, bool)``.

        Parameters
        ----------
        depth : Frame
            ``(512, 424)`` depth frame

        undistorted : Frame
            ``(512, 424)`` registered depth frame

        Raises
        ------
        ValueError
            If invalid shape of frame/array is provided
        """

        if depth.width != 512 or depth.height != 424:
            raise ValueError(
                "Shape of depth frame {}x{} is invalid. Expected 512x424.".format(
                depth.width, depth.height))
        if undistorted.width != 512 or undistorted.height != 424:
            raise ValueError(
                "Shape of undistorted frame {}x{} is invalid. Expected 512x424.".format(
                undistorted.width, undistorted.height))
        self.ptr.undistortDepth(depth.ptr, undistorted.ptr)

    def getPointXYZRGB(self, Frame undistorted, Frame registered, r, c):
        """Same as ``libfreenect2::Registration::getPointXYZRGB``.

        Parameters
        ----------
        undistorted : Frame
            ``(512, 424)`` Undistorted depth frame

        registered : Frame
            ``(512, 424)`` Registered color frame

        r : int
            Row (y) index in depth image

        c : int
            Column (x) index in depth image.

        Returns
        -------
        tuple : (X coordinate of the 3-D point (meter),
                 Y coordinate of the 3-D point (meter),
                 Z coordinate of the 3-D point (meter),
                 B,
                 G,
                 R)

        """
        cdef float x = 0, y = 0, z = 0, rgb = 0
        self.ptr.getPointXYZRGB(undistorted.ptr, registered.ptr, r, c, x, y, z, rgb)
        cdef uint8_t* bgrptr = reinterpret_cast[uint8_pt](&rgb);
        return (x, y, z, bgrptr[0], bgrptr[1], bgrptr[2])

    def getPointXYZ(self, Frame undistorted, r, c):
        """Same as ``libfreenect2::Registration::getPointXYZ``.

        Parameters
        ----------
        undistorted : Frame
            ``(512, 424)`` Undistorted depth frame

        r : int
            Row (y) index in depth image

        c : int
            Column (x) index in depth image.

        Returns
        -------
        tuple : (X coordinate of the 3-D point (meter),
                 Y coordinate of the 3-D point (meter),
                 Z coordinate of the 3-D point (meter))

        """
        cdef float x = 0, y = 0, z = 0
        self.ptr.getPointXYZ(undistorted.ptr, r, c, x, y, z)
        return (x, y, z)


cdef class Logger:
    """Python interface for libfreenect2::Logger


    The Logger is a container of C++ pointer ``libfreenect2::Logger*``.

    Attributes
    ----------
    ptr : libfreenect2::Logger*
        C++ pointer of Logger

    """
    cdef libfreenect2.Logger* ptr

    def level(self):
        """Same as ``Level level()``"""
        return self.ptr.level()

    def log(self, level, message):
        """Same as ``void log(Level level, const std::string &messagge)``"""
        self.ptr.log(level, message)


def createConsoleLogger(level):
    """Same as ``Logger* libfreenect2::createConsoleLogger(Level level)``

    Parameters
    ----------
    level : pylibfreenect2.LoggerLevel
        Logger level

    Returns
    -------
    logger : Logger
        Allocated logger

    Examples
    --------
    .. code-block:: python

        logger = pylibfreenect2.createConsoleLogger(
            pylibfreenect2.LoggerLevel.Debug)

    See also
    --------

    pylibfreenect2.LoggerLevel
    pylibfreenect2.libfreenect2.createConsoleLoggerWithDefaultLevel
    """
    cdef Logger logger = Logger()
    logger.ptr = libfreenect2.createConsoleLogger(level)
    return logger


def createConsoleLoggerWithDefaultLevel():
    """Same as ``Logger* libfreenect2::createConsoleLoggerWithDefaultLevel()``

    Returns
    -------
    logger : Logger
        Allocated logger

    See also
    --------

    pylibfreenect2.libfreenect2.createConsoleLogger
    """
    cdef Logger logger = Logger()
    cdef libfreenect2.Logger* ptr = libfreenect2.createConsoleLoggerWithDefaultLevel()
    logger.ptr = <libfreenect2.Logger*>ptr
    return logger


def getGlobalLogger():
    """Same as ``Logger* libfreenect2::getGlobalLogger()``"""
    cdef Logger logger = Logger()
    logger.ptr = libfreenect2.getGlobalLogger()
    return logger


def setGlobalLogger(Logger logger=None):
    """Same as ``void libfreenect2::getGlobalLogger(Logger*)``

    Parameters
    ----------
    logger : Logger
        Python side instance for ``libfreenect2::Logger*``. If None,
        ``setGlobalLogger(NULL)`` will be called, i.e. logging is disabled.
        Default is None.

    """
    if logger is None:
        libfreenect2.setGlobalLogger(NULL)
    else:
        libfreenect2.setGlobalLogger(logger.ptr)


# MUST be declared before backend specific includes
cdef class PacketPipeline:
    """Base class for other pipeline classes.

    Attributes
    ----------
    pipeline_ptr_alias : ``libfreenect2::PacketPipeline*``
    owened_by_device : bool

    See also
    --------
    pylibfreenect2.libfreenect2.CpuPacketPipeline
    pylibfreenect2.libfreenect2.OpenCLPacketPipeline
    pylibfreenect2.libfreenect2.OpenCLKdePacketPipeline
    pylibfreenect2.libfreenect2.OpenGLPacketPipeline
    """
    cdef libfreenect2.PacketPipeline* pipeline_ptr_alias

    # NOTE: once device is opened with pipeline, pipeline will be
    # releaseed in the destructor of Freenect2DeviceImpl
    cdef bool owned_by_device


cdef class CpuPacketPipeline(PacketPipeline):
    """Pipeline with CPU depth processing.

    Attributes
    ----------
    pipeline : `libfreenect2::CpuPacketPipeline*`
    """
    cdef libfreenect2.CpuPacketPipeline* pipeline

    def __cinit__(self):
        self.pipeline = new libfreenect2.CpuPacketPipeline()
        self.pipeline_ptr_alias = <libfreenect2.PacketPipeline*>self.pipeline
        self.owned_by_device = False

    def __dealloc__(self):
        if not self.owned_by_device:
            if self.pipeline is not NULL:
                del self.pipeline

IF LIBFREENECT2_WITH_OPENGL_SUPPORT == True:
    include "opengl_packet_pipeline.pxi"

IF LIBFREENECT2_WITH_OPENCL_SUPPORT == True:
    include "opencl_packet_pipeline.pxi"
    # include "opencl_kde_packet_pipeline.pxi"

cdef class Freenect2Device:
    """Python interface for ``libfreenect2::Freenect2Device``.

    The Freenect2Device is a container of C++ pointer
    ``libfreenect2::Freenect2Device*``.

    .. note::
        Freenect2Device just keeps a pointer of
        ``libfreenect2::Freenect2Device`` that should be allocated and released
        by Freenect2. Freenect2Device itself doesn't own the memory.


    A valid device can be created by ``openDefaultDevice``:

    .. code-block:: python

        fn = Freenect2()
        assert fn.enumerateDevices() > 0
        device = fn.openDefaultDevice()

    or  by ``openDevice``:

    .. code-block:: python

        fn = Freenect2()
        assert fn.enumerateDevices() > 0
        serial = fn.getDeviceSerialNumber(0)
        device = fn.openDevice(serial)

    Attributes
    ----------
    ptr : ``libfreenect2::Freenect2Device*``

    See also
    --------
    pylibfreenect2.libfreenect2.Freenect2
    pylibfreenect2.libfreenect2.Freenect2.openDefaultDevice
    pylibfreenect2.libfreenect2.Freenect2.openDevice

    """

    cdef _Freenect2Device* ptr

    def getSerialNumber(self):
        """Same as ``libfreenect2::Freenect2Device::getSerialNumber()``"""
        return self.ptr.getSerialNumber()

    def getFirmwareVersion(self):
        """Same as ``libfreenect2::Freenect2Device::getFirmwareVersion()``"""
        return self.ptr.getFirmwareVersion()

    def getColorCameraParams(self):
        """Same as ``libfreenect2::Freenect2Device::getColorCameraParams()``"""
        cdef _Freenect2Device.ColorCameraParams params
        params = self.ptr.getColorCameraParams()
        cdef ColorCameraParams pyparams = ColorCameraParams()
        pyparams.params = params
        return pyparams

    def getIrCameraParams(self):
        """Same as ``libfreenect2::Freenect2Device::getIrCameraParams()``"""
        cdef _Freenect2Device.IrCameraParams params
        params = self.ptr.getIrCameraParams()
        cdef IrCameraParams pyparams = IrCameraParams()
        pyparams.params = params
        return pyparams

    def setColorFrameListener(self, FrameListener listener):
        """Same as
        ``libfreenect2::Freenect2Device::setColorFrameListener(FrameListener*)``
        """
        self.ptr.setColorFrameListener(listener.listener_ptr_alias)

    def setIrAndDepthFrameListener(self, FrameListener listener):
        """Same as
        ``libfreenect2::Freenect2Device::setIrAndDepthFrameListener(FrameListener*)``
        """
        self.ptr.setIrAndDepthFrameListener(listener.listener_ptr_alias)

    def start(self):
        """Same as ``libfreenect2::Freenect2Device::start()``"""
        self.ptr.start()

    def startStreams(self, rgb, depth):
        """Same as ``libfreenect2::Freenect2Device::startStreams(bool, bool)``"""
        self.ptr.startStreams(rgb, depth)

    def stop(self):
        """Same as ``libfreenect2::Freenect2Device::stop()``"""
        self.ptr.stop()

    def close(self):
        """Same as ``libfreenect2::Freenect2Device::close()``"""
        self.ptr.close()


cdef class Freenect2:
    """Python interface for ``libfreenect2::Freenect2``.

    The Freenect2 is a container of C++ pointer
    ``libfreenect2::Freenect2*``. The pointer of Freenect2 is allocated
    in ``__cinit__`` and released in ``__dealloc__`` method.

    Attributes
    ----------
    ptr : ``libfreenect2::Freenect2*``

    See also
    --------
    pylibfreenect2.libfreenect2.Freenect2Device

    """

    cdef libfreenect2.Freenect2* ptr

    def __cinit__(self):
        self.ptr = new libfreenect2.Freenect2();

    def __dealloc__(self):
        if self.ptr is not NULL:
            del self.ptr

    def enumerateDevices(self):
        """Same as ``libfreenect2::Freenect2::enumerateDevices()``"""
        return self.ptr.enumerateDevices()

    def getDeviceSerialNumber(self, int idx):
        """Same as ``libfreenect2::Freenect2::getDeviceSerialNumber(int)``"""
        return self.ptr.getDeviceSerialNumber(idx)

    def getDefaultDeviceSerialNumber(self):
        """Same as ``libfreenect2::Freenect2::getDefaultDeviceSerialNumber()``"""
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
        """Open device by serial number or index

        Parameters
        ----------
        name : int or str
            Serial number (str) or device index (int)

        pipeline : PacketPipeline, optional
            Pipeline. Default is None.

        Raises
        ------
        ValueError
            If invalid name is specified.

        """
        if isinstance(name, int):
            return self.__openDevice__intidx(name, pipeline)
        elif isinstance(name, str) or isinstance(name, bytes):
            return self.__openDevice__stridx(name, pipeline)
        else:
            raise ValueError("device name must be of str, bytes or integer type")

    def openDefaultDevice(self, PacketPipeline pipeline=None):
        """Open the first device

        Parameters
        ----------
        pipeline : PacketPipeline, optional
            Pipeline. Default is None.

        Returns
        -------
        device : Freenect2Device

        """
        cdef _Freenect2Device* dev_ptr

        if pipeline is None:
            dev_ptr = self.ptr.openDefaultDevice()
        else:
            dev_ptr = self.ptr.openDefaultDevice(pipeline.pipeline_ptr_alias)
            pipeline.owned_by_device = True

        cdef Freenect2Device device = Freenect2Device()
        device.ptr = dev_ptr
        return device
