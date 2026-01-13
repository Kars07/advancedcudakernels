#include <cuda.h>

__constant__ CUtensorMap global_tensor_map;
__global__ void kernel() {
    // Use global_tensor_map here
}

int main() {
    CUtensorMap local_tensor_map;
    // [ ..Initialize map]
    cudaMemcpyToSymbol(global_tensor_map, &local_tensor_map, sizeof(CUtensorMap));
    kernel<<<1, 1>>>();
}
