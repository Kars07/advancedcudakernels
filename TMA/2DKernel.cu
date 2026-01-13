#include <cuda.h> // CUtensormap
#include <cuda/barrier>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;


__global__ void  kernel(const __grid_constant__ CUtensorMap tensor_map, int x, int y) {
    //The destination shared memory buffer of a bulk tensor operation should
    // be 128 byte aligned
    __shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

    // Initialize shared memory buffer with the number of threads participating
    // in the barrier
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

    if (threadIdx.x == 0) {
        // Initialize barrier. All `blockDim.x` threads in block participate
        init(&bar, blockDim.x);
        // Make initialized barrier visible in async proxy.
        cde::fence_proxy_async_shared_cta();
    }
    // Syncthreads so initialized barrier is visible to all threads
    __syncthreads();

    barrier::arrival_token token;
    if (threadIdx.x == 0) {
        // Initiate bulk tensor copy
        cde::cp_async_bulk_tensor_2d_global_to_shared(&smem_buffer, &tensor_map, x, y,￿ bar);
        // Arrive on the barrier and tell how many bytes are expected to come in.
        token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
    } else {
        // Other threads just arrive
        token = bar.arrive();
    }
    // Wait for the data to have arrived
    bar.wait(std::move(token));

    // Symbolically modify a value in shared memory.
    smem_buffer[0][threadIdx.x] += threadIdx.x;

    // Wait for shared memory writes to be visible to TMA engine.
    cde::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, x, y, &smem_buffer);
        // Wait for TMA transfer to have finished reading shared memory.
        // Create a "bulk async-group" out of the previous bulk copy operation.
        cde::cp_async_bulk_commit_group();
        // Wait for the group to have completed reading from shared memory.
        cde::cp_async_bulk_wait_group_read<0>();
    }

    // Destroy barrier. This invalidates the memory region of the barrier. If
    // further computations were to take place in the kernel, this allows the
    // memory location of the shared memory barrier to be reused.
    if (threadIdx.x == 0) {
        (&bar)->~barrier();
    }
}

/*
Negative indices and out of bounds. When part of the tile that is being read from global to shared
memory is out of bounds, the shared memory that corresponds to the out of bounds area is zerofilled.
The top-left corner indices of the tile may also be negative. When writing from shared to global
memory, parts of the tile may be out of bounds, but the top left corner cannot have any negative
indices.

Size and stride. The size of a tensor is the number of elements along one dimension. All sizes must
be greater than one. The stride is the number of bytes between elements of the same dimension. For
instance, a 4 x 4 matrix of integers has sizes 4 and 4. Since it has 4 bytes per element, the strides are 4
and 16 bytes. Due to alignment requirements, a 4 x 3 row-major matrix of integers must have strides of
4 and 16 bytes as well. Each row is padded with 4 extra bytes to ensure that the start of the next row is
aligned to 16 bytes.

Address / Size Alignment
Global memory address Must be 16 byte aligned.
Global memory sizes Must be greater than or equal to one. Does not have to be a multiple
of 16 bytes.
Global memory strides Must be multiples of 16 bytes.
Shared memory address Must be 128 byte aligned.
Shared memory barrier address
Must be 8 byte aligned (this is guaranteed by cuda::barrier).
Size of transfer Must be a multiple of 16 bytes.
*/
