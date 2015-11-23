cdef class OpenCLPacketPipeline(PacketPipeline):
    cdef _OpenCLPacketPipeline* pipeline

    def __cinit__(self, int device_id=-1):
        self.pipeline = new _OpenCLPacketPipeline(device_id)
        self.pipeline_ptr_alias = <_PacketPipeline*>self.pipeline
        self.owned_by_device = False

    def __dealloc__(self):
        if not self.owned_by_device:
            if self.pipeline is not NULL:
                del self.pipeline
