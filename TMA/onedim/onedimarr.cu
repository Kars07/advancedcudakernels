/*
This section demonstrates how to write a simple kernel that read-modify-writes a one-dimensional
array using TMA. This shows how to how to load and store data using bulk-asynchronous copies, as
well as how to synchronize threads of execution with those copies.
The code of the kernel is included below. Some functionality requires inline PTX assembly that is currently
made available through libcu++. The availability of these wrappers can be checked with the
following code:

#if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 900
static_assert(false, "Device code is being compiled with older architectures that are￿
,→incompatible with TMA.");
#endif ∕∕ __CUDA_MINIMUM_ARCH__

The kernel goes through the following stages:
    1. Initialize shared memory barrier.
    2. Initiate bulk-asynchronous copy of a block of memory from global to shared memory.
    3. Arrive and wait on the shared memory barrier.
    4. Increment the shared memory buffer values.
    5. Wait for shared memory writes to be visible to the subsequent bulk-asynchronous copy, i.e., order
    the shared memory writes in the async proxy before the next step.
    6. Initiate bulk-asynchronous copy of the buffer in shared memory to global memory.
    7. Wait at end of kernel for bulk-asynchronous copy to have finished reading shared memory.
*/

#include <cuda/barrier>
#include <cuda/ptx>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

static constexpr size_t buf_len = 1024;
__global__ void add_one_kernel(int* data, size_t offset){
    // Shared memory buffer. The destination shared memory buffer of
    // a bulk operations should be 16 byte aligned.
    __shared__ alignas(16) int smem_data[buf_len];

    // 1. a) Initialize shared memory barrier with the number of threads participating in￿ the barrier.
    //    b) Make initialized barrier visible in async proxy.
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;
    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);                     // a)
        ptx::fence_proxy_async(ptx::space_shared);  // b)
    }
    __syncthreads();


    // 2. Initiate TMA transfer to copy global to shared memory.
    if (threadIdx.x == 0) {
        // 3a. cuda::memcpy_async arrives on the barrier and communicates
        // how many bytes are expected to come in (the transaction count)
      cuda::memcpy_async(
        smem_data,
        data + offset,
        cuda::aligned_size_t<16>(sizeof(smem_data)),
        bar
      );
    }
    // 3b. All threads arrive on the barrier
    barrier::arrival_token token = bar.arrive();

    // 3c. Wait for the data to have arrived.
    bar.wait(std::move(token));

    // 4. Compute saxpy and write back to shared memory
    for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
        smem_data[i] += 1;
    }

    // 5. Wait for shared memory writes to be visible to TMA engine.
    ptx::fence_proxy_async(ptx::space_shared); // b)
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.
    //
    // 6. Initiate TMA transfer to copy shared memory to global memory
    if (threadIdx.x == 0) {
        ptx::cp_async_bulk(
            ptx::space_global,
            ptx::space_shared,
            data + offset, smem_data, sizeof(smem_data));
        // 7. Wait for TMA transfer to have finished reading shared memory.
        // Create a "bulk async-group" out of the previous bulk copy operation.
        ptx::cp_async_bulk_commit_group();
        // Wait for the group to have completed reading from shared memory.
        ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
    }
}
