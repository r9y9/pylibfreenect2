cdef class OpenGLPacketPipeline(PacketPipeline):
    cdef _OpenGLPacketPipeline* pipeline

    def __cinit__(self, bool debug=False):
        self.pipeline = new _OpenGLPacketPipeline(NULL, debug)
        self.pipeline_ptr_alias = <_PacketPipeline*>self.pipeline

    def __dealloc__(self):
        pass
