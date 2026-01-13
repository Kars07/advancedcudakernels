/*
The primary difference between the one-dimensional and multi-dimensional case is that a tensor map
must be created on the host and passed to the CUDA kernel. This section describes how to create a
tensor map using the CUDA driver API, how to pass it to device, and how to use it on device.
Driver API. A tensor map is created using the cuTensorMapEncodeTiled driver API. This API can be
accessed by linking to the driver directly (-lcuda) or by using the cudaGetDriverEntryPoint API. Below,
we show how to get a pointer to the cuTensorMapEncodeTiled API. For more information, refer to
Driver Entry Point Access.
*/

#include <cstdint>
#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
    // Get a pointer to cuTensorMapEncodeTiled
    cudaDriverEntryPointQueryResult driver_status;
    void* cuTensorMapEncodeTiled_ptr = nullptr;
    CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault, &driver_status));
    assert(driver_status == cudaDriverEntryPointSuccess);

    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}

/*
Creation. Creating a tensor map requires many parameters. Among them are the base pointer to an
array in global memory, the size of the array (in number of elements), the stride from one row to the
next (in bytes), the size of the shared memory buffer (in number of elements). The code below creates
a tensor map to describe a two-dimensional row-major array of size GMEM_HEIGHT x GMEM_WIDTH.
Note the order of the parameters: the fastest moving dimension comes first.
*/

CUtensorMap tensor_map();
// rank is the number of dimensions of the array
constexpr uint32_t rank = 2;
uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};
// The stride is the number of bytes to traverse from the first element
// of one row to the next
// it must be a multiply of 16
uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};
// The box size is the size of the shared memory buffer that is used
// as the destination of a TMA transfer
uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};
// The distance between elements in units of sizeof(element). A stride of 2
// can be used to load only the real component of a complex-valued tensor, for instance
uint32_t elem_stride[rank] = {1, 1};

// Get a function pointer to the cuTensorMapEncodeTiled driver API.
auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

// Create the tensor descriptor
CUresult res = cuTensorMapEncodeTiled (
    &tensor_map, // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
    rank, // cuuint32_t tensorRank,
    tensor_ptr, // void *globalAddress,
    size, // const cuuint64_t *globalDim,
    stride, // const cuuint64_t *globalStrides,
    box_size, // const cuuint32_t *boxDim,
    elem_stride, // const cuuint32_t *elementStrides,
    // Interleave patterns can be used to accelerate loading of values that
    // are less than 4 bytes long.
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    // Swizzling can be used to avoid shared memory bank conflicts.
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    // L2 Promotion can be used to widen the effect of a cache-policy to a wider
    // set of L2 cache lines.
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    // Any element that is outside of bounds will be set to zero
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);
