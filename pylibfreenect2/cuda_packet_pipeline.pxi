cdef class CudaPacketPipeline(PacketPipeline):
    """Pipeline with Cuda depth processing.

    Attributes
    ----------
    pipeline : `libfreenect2::CudaPacketPipeline*`

    Parameters
    ----------
    device_id : int, optional
        Device id. Default is -1.

    """
    cdef libfreenect2.CudaPacketPipeline * pipeline

    def __cinit__(self, int device_id=-1):
        self.pipeline = new libfreenect2.CudaPacketPipeline(device_id)
        self.pipeline_ptr_alias = <libfreenect2.PacketPipeline * >self.pipeline
        self.owned_by_device = False

    def __dealloc__(self):
        if not self.owned_by_device:
            if self.pipeline is not NULL:
                del self.pipeline
