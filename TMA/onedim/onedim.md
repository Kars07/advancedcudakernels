Barrier initialization. The barrier is initialized with the number of threads participating in the block. As a result, the barrier will flip only if all threads have arrived on this barrier. To make the initialized barrier visible to subsequent bulk-asynchronous copies, the fence.proxy.async.shared::cta instruction is used. This instruction ensures that subsequent bulk-asynchronous copy operations operate on the initialized barrier.
TMA read. The bulk-asynchronous copy instruction directs the hardware to copy a large chunk of
data into shared memory, and to update the transaction count of the shared memory barrier after
completing the read. In general, issuing as few bulk copies with as big a size as possible results in the best performance. Because the copy can be performed asynchronously by the hardware, it is not
necessary to split the copy into smaller chunks.
The thread that initiates the bulk-asynchronous copy operation arrives at the barrier using mbarrier.
expect_tx. This is automatically performed by cuda::memcpy_async. This tells the barrier that
the thread has arrived and also how many bytes (tx / transactions) are expected to arrive. Only a
single thread has to update the expected transaction count. If multiple threads update the transaction
count, the expected transaction will be the sum of the updates. The barrier will only flip once all
threads have arrived and all bytes have arrived. Once the barrier has flipped, the bytes are safe to read
from shared memory, both by the threads as well as by subsequent bulk-asynchronous copies.
Barrier wait. Waiting for the barrier to flip is done using mbarrier.try_wait. It can either return
true, indicating that the wait is over, or return false, which may mean that the wait timed out. The
while loop waits for completion, and retries on time-out.
SMEM write and sync. The increment of the buffer values reads and writes to shared memory. To make
the writes visible to subsequent bulk-asynchronous copies, the fence.proxy.async.shared::cta
instruction is used. This orders the writes to shared memory before subsequent reads from bulkasynchronous
copy operations, which read through the async proxy. So each thread first orders the
writes to objects in shared memory in the async proxy via the fence.proxy.async.shared::cta,
and these operations by all threads are ordered before the async operation performed in thread 0
using __syncthreads().
TMA write and sync. The write from shared to global memory is again initiated by a single thread. The
completion of the write is not tracked by a shared memory barrier. Instead, a thread-local mechanism
is used. Multiple writes can be batched into a so-called bulk async-group. Afterwards, the thread can
wait for all operations in this group to have completed reading from shared memory (as in the code
above) or to have completed writing to global memory, making the writes visible to the initiating thread.
For more information, refer to the PTX ISA documentation of cp.async.bulk.wait_group. Note that the
bulk-asynchronous and non-bulk asynchronous copy instructions have different async-groups: there
exist both cp.async.wait_group and cp.async.bulk.wait_group instructions.



Address/Size       Alignment
Address / Size Alignment
Global memory address Must be 16 byte aligned.
Shared memory address Must be 16 byte aligned.
Shared memory barrier address Must be 8 byte aligned (this is guaranteed by cuda::barrier).
Size of transfer Must be a multiple of 16 bytes.
