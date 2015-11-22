cdef class OpenCLPacketPipeline(PacketPipeline):
    cdef _OpenCLPacketPipeline* pipeline

    def __cinit__(self, int device_id=-1):
        self.pipeline = new _OpenCLPacketPipeline(device_id)
        self.pipeline_ptr_alias = <_PacketPipeline*>self.pipeline

    def __dealloc__(self):
        pass
