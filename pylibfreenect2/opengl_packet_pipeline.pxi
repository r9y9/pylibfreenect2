cdef class OpenGLPacketPipeline(PacketPipeline):
    cdef _OpenGLPacketPipeline* pipeline

    def __cinit__(self, bool debug=False):
        self.pipeline = new _OpenGLPacketPipeline(NULL, debug)
        self.pipeline_ptr_alias = <_PacketPipeline*>self.pipeline
        self.owned_by_device = False

    def __dealloc__(self):
        if not self.owned_by_device:
            if self.pipeline is not NULL:
                del self.pipeline
