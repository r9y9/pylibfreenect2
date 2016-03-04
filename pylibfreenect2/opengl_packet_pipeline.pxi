cdef class OpenGLPacketPipeline(PacketPipeline):
    """Pipeline with OpenGL depth processing.

    Attributes
    ----------
    pipeline : `libfreenect2::OpenGLPacketPipeline*`

    Parameters
    ----------
    debug : bool, optional
        Debugging mode. Default is False.

    """
    cdef libfreenect2.OpenGLPacketPipeline* pipeline

    def __cinit__(self, bool debug=False):
        self.pipeline = new libfreenect2.OpenGLPacketPipeline(NULL, debug)
        self.pipeline_ptr_alias = <libfreenect2.PacketPipeline*>self.pipeline
        self.owned_by_device = False

    def __dealloc__(self):
        if not self.owned_by_device:
            if self.pipeline is not NULL:
                del self.pipeline
