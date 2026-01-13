/*
Host-to-device transfer. There are three ways to make a tensor map accessible to device code. The
recommended approach is to pass the tensor map as a const __grid_constant__ parameter to a
kernel. The other possibilities are copying the tensor map into device __constant__ memory using
cudaMemcpyToSymbol or accessing it via global memory. When passing the tensor map as a parameter,
some versions of the GCC C++ compiler issue the warning “the ABI for passing parameters with
64-byte alignment has changed in GCC 4.6”. This warning can be ignored.
*/

#include <__clang_cuda_runtime_wrapper.h>
#include <cuda.h>

__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map) {
    // Use tensor_map here
}

int main() {
    CUtensorMap map;
    // [ ..Initialize map..]
    kernel<<<1, 1>>>(map);
}
