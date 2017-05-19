cdef class OpenCLKdePacketPipeline(PacketPipeline):
    """Depth packet processing pipeline using KDE depth unwrapping algorithms

    Attributes
    ----------
    pipeline : `libfreenect2::OpenCLKdePacketPipeline*`

    Parameters
    ----------
    device_id : int, optional
        Device id. Default is -1.

    References
    ----------
    https://github.com/OpenKinect/libfreenect2/pull/742

    """
    cdef libfreenect2.OpenCLKdePacketPipeline * pipeline

    def __cinit__(self, int device_id=-1):
        self.pipeline = new libfreenect2.OpenCLKdePacketPipeline(device_id)
        self.pipeline_ptr_alias = <libfreenect2.PacketPipeline * >self.pipeline
        self.owned_by_device = False

    def __dealloc__(self):
        if not self.owned_by_device:
            if self.pipeline is not NULL:
                del self.pipeline
