
// Consumer of tensor map in global memory:
__global__ void consume_tensor_map(CUtensorMap* tensor_map) {
// Fence acquire tensor map:
ptx::n32_t<128> size_bytes;
ptx::fence_proxy_tensormap_generic(ptx::sem_acquire, ptx::scope_sys, tensor_map, size_bytes);
// Safe to use tensor_map after fence..
__shared__ uint64_t bar;
__shared__ alignas(128) char smem_buf[4][128];
if (threadIdx.x == 0) {
// Initialize barrier
ptx::mbarrier_init(&bar, 1);
// Make barrier init visible in async proxy, i.e., to TMA engine
ptx::fence_proxy_async(ptx::space_shared);
// Issue TMA request
ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global, smem_buf, tensor_map, {0, 0}, &bar);
// Arrive on barrier. Expect 4 * 128 bytes.
ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar, sizeof(smem_buf));
}
const int parity = 0;
// Wait for load to have completed
while (!ptx::mbarrier_try_wait_parity(&bar, parity)) {}
// print items:
printf("Got:\n\n");
for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 128; ++i) {
        printf("%3d ", smem_buf[j][i]);
        if (i % 32 == 31) { printf("\n"); };
    }
    printf("\n");
    }
}
