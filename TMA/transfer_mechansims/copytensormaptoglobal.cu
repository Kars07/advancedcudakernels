/*
Finally, it is possible to copy the tensor map to global memory. Using a pointer to a tensor map in global
device memory requires a fence in each thread block before any thread in the block uses the updated
tensor map. Further uses of the tensor map by that thread block do not need to be fenced unless
the tensor map is modified again. Note that this mechanism may be slower than the two mechanisms.
*/


#include <cuda.h>
#include <cuda/ptx>
namespace ptx = cuda::ptx;

__device__ CUtensorMap global_tensor_map;
__global__ void kernel(CUtensorMap *tensor_map)
{
// Fence acquire tensor map:
ptx::n32_t<128> size_bytes;
// Since the tensor map was modified from the host using cudaMemcpy,
// the scope should be .sys.
ptx::fence_proxy_tensormap_generic(
ptx::sem_acquire, ptx::scope_sys, tensor_map, size_bytes
);
// Safe to use tensor_map after fence inside this thread..
}
int main() {
CUtensorMap local_tensor_map;
// [ ..Initialize map.. ]
cudaMemcpy(&global_tensor_map, &local_tensor_map, sizeof(CUtensorMap),￿
cudaMemcpyHostToDevice);
kernel<<<1, 1>>>(global_tensor_map);
}
