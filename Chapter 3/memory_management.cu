#include <cstdio>
#include <cuda_runtime.h>

// -----------------------------
// WHY: Demonstrate different memory types and their characteristics.
// Global memory (passed as pointer), shared memory (declared inside),
// and registers (local variables) all behave differently in terms of
// latency, scope, and lifetime.
// -----------------------------
__global__ void memory_hierarchy_demo(
    int *global_in,      // WHY: Lives in global DRAM (slow, large, persistent)
    int *global_out,     // WHY: Output buffer in global memory
    int N)               // WHY: Problem size passed as parameter
{
    // -----------------------------
    // Thread identity
    // WHY: Same as Chapter 2—we need to know which element each thread owns
    // and its position within the block for shared memory indexing.
    // -----------------------------
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx  = threadIdx.x;

    // -----------------------------
    // Guard against over-launch
    // WHY: Prevent out-of-bounds access when grid size exceeds N.
    // -----------------------------
    if (global_idx >= N) return;

    // -----------------------------
    // Shared memory allocation
    // WHY: Shared memory is ON-CHIP (fast, ~20-30 cycles) but LIMITED
    // (~48-164KB per SM). We allocate it per-block, not per-thread.
    // All threads in THIS BLOCK can access this array. It only exists
    // during block execution—destroyed when block finishes.
    // We use blockDim.x (block size) as the size because each thread
    // will store one value. Static size must be known at compile time.
    // -----------------------------
    __shared__ int shared_data[256];  // Assuming max block size of 256

    // -----------------------------
    // Register variables
    // WHY: Local variables live in REGISTERS (fastest, 1-cycle access).
    // Each thread gets its own private copy—no sharing between threads.
    // Registers are THE most limited resource (32K-64K per SM total).
    // Using too many registers reduces occupancy because fewer threads
    // can be resident simultaneously. The compiler allocates these
    // automatically from the thread's register file.
    // -----------------------------
    int register_value = global_idx * 2;
    int temp_sum = 0;

    // -----------------------------
    // COALESCED global memory read
    // WHY: Threads in a warp (0-31) access consecutive addresses:
    // Thread 0 reads global_in[0], Thread 1 reads global_in[1], etc.
    // The hardware COMBINES these 32 individual 4-byte reads into a
    // single 128-byte memory transaction. This is COALESCED access—
    // the most efficient pattern. Without coalescing, this could be
    // 32 separate transactions, wasting 31/32 of the bandwidth.
    // We load from slow global memory (400+ cycle latency) but the
    // SM will context-switch to other warps while waiting, hiding latency.
    // -----------------------------
    int loaded_value = global_in[global_idx];

    // -----------------------------
    // Write to shared memory
    // WHY: Each thread stores its loaded value in shared memory at its
    // local index. This is a BROADCAST pattern—each thread writes to a
    // different location, no conflicts. Shared memory is organized into
    // 32 BANKS. Since we're writing shared_data[threadIdx.x] with
    // consecutive threadIdx values, each thread accesses a different bank
    // (bank = address % 32), so all writes happen in PARALLEL.
    // Without the barrier below, reading shared_data[other_index] would
    // race—we might read before another thread has written.
    // -----------------------------
    shared_data[local_idx] = loaded_value;

    // -----------------------------
    // Block-level synchronization barrier
    // WHY: CRITICAL for shared memory correctness. Before this point,
    // threads are executing independently—thread 0 might write to
    // shared_data[0] while thread 31 hasn't written to shared_data[31] yet.
    // __syncthreads() BLOCKS all threads in this block until ALL reach
    // this point. After the barrier, we're GUARANTEED that all threads
    // have completed their writes to shared_data. Without this, the
    // reduction loop below would read garbage or stale data.
    // This is implemented in hardware using per-SM synchronization counters.
    // -----------------------------
    __syncthreads();

    // -----------------------------
    // Reduction: sum values from shared memory
    // WHY: Demonstrate DATA REUSE—the key reason shared memory exists.
    // We loaded from global memory ONCE (slow), now we can read from
    // shared memory MANY TIMES (fast). Each thread reads multiple values
    // written by OTHER threads. Without shared memory, we'd need to
    // re-read from global memory 256 times, wasting bandwidth.
    // This creates BANK CONFLICTS: multiple threads in a warp may access
    // the same bank (address % 32) with different addresses, serializing
    // access. But for a simple reduction, the performance is still better
    // than global memory.
    // -----------------------------
    for (int i = 0; i < blockDim.x; i++) {
        temp_sum += shared_data[i];
    }

    // -----------------------------
    // Compute using register values
    // WHY: All computation here uses REGISTERS (temp_sum, register_value).
    // Registers are 1-cycle access, so these operations are essentially free
    // from a memory perspective. This demonstrates REGISTER-BOUND computation.
    // The more we can keep data in registers, the faster the kernel runs.
    // But we're LIMITED—if we declare too many local variables, they spill
    // to "local memory" (actually cached global memory), killing performance.
    // -----------------------------
    int result = temp_sum + register_value;

    // -----------------------------
    // Another synchronization barrier
    // WHY: Not strictly necessary here since we're about to write to
    // independent global memory locations, but demonstrates that barriers
    // can be used at ANY point where threads need to coordinate.
    // Common pattern: compute → sync → compute → sync → write.
    // If we were writing BACK to shared_data, this barrier would be
    // CRITICAL to prevent threads from overwriting data other threads
    // are still reading.
    // -----------------------------
    __syncthreads();

    // -----------------------------
    // COALESCED global memory write
    // WHY: Same coalescing principle as the read above. Threads write to
    // consecutive addresses (global_out[0], global_out[1], ...). The
    // hardware combines these into efficient memory transactions.
    // If we wrote global_out[global_idx * 2] (stride-2 access), we'd
    // waste half the bandwidth. If we wrote global_out[random_index],
    // we'd potentially issue 32 separate transactions—32x slower.
    // Global memory writes are ASYNCHRONOUS—they don't block the thread.
    // The write is posted to the memory controller and the thread continues.
    // There's NO GUARANTEE about ordering between threads unless you
    // use atomics or explicit synchronization.
    // -----------------------------
    global_out[global_idx] = result;
}

// -----------------------------
// WHY: Demonstrate UNCOALESCED access pattern to show performance impact.
// This kernel intentionally uses a stride-2 pattern to show what happens
// when memory accesses are NOT coalesced.
// -----------------------------
__global__ void uncoalesced_access_demo(int *data, int N)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (global_idx >= N / 2) return;  // WHY: Only process half the elements

    // -----------------------------
    // UNCOALESCED: stride-2 access pattern
    // WHY: Thread 0 accesses data[0], Thread 1 accesses data[2],
    // Thread 2 accesses data[4], etc. Within a warp (32 threads),
    // we're accessing 64 different addresses spread across multiple
    // cache lines. The hardware CANNOT combine these into a single
    // transaction. Instead of 1 transaction, we get MULTIPLE transactions,
    // each fetching a 128-byte cache line but only using half the data.
    // Result: ~2x bandwidth waste compared to coalesced access.
    // This is a COMMON performance bug in real kernels.
    // -----------------------------
    int stride_idx = global_idx * 2;
    data[stride_idx] = global_idx;
}

// -----------------------------
// WHY: Demonstrate shared memory BANK CONFLICTS.
// Bank conflicts serialize access within a warp, similar to warp divergence
// but for memory instead of control flow.
// -----------------------------
__global__ void bank_conflict_demo(int *output, int N)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx  = threadIdx.x;
    
    if (global_idx >= N) return;

    __shared__ int shared_data[256];

    // -----------------------------
    // NO CONFLICT: each thread accesses different bank
    // WHY: Consecutive threadIdx values access consecutive addresses.
    // Shared memory banks are assigned address % 32, so threadIdx 0 → bank 0,
    // threadIdx 1 → bank 1, ..., threadIdx 31 → bank 31. All parallel.
    // -----------------------------
    shared_data[local_idx] = global_idx;
    __syncthreads();

    // -----------------------------
    // BANK CONFLICT: stride-2 causes same bank, different address
    // WHY: Thread 0 accesses address 0 (bank 0), Thread 1 accesses address 2
    // (bank 2), ..., Thread 16 accesses address 32 (bank 0 again!).
    // Now threads 0 and 16 both want bank 0 simultaneously—CONFLICT.
    // These accesses SERIALIZE: first thread 0 reads (thread 16 waits),
    // then thread 16 reads (thread 0 waits). Same for all other bank pairs.
    // Result: ~2x slower than no-conflict access. With stride-32, ALL threads
    // in a warp hit the same bank → 32x serialization (worst case).
    // -----------------------------
    int conflict_idx = (local_idx * 2) % blockDim.x;
    int value = shared_data[conflict_idx];
    
    __syncthreads();

    output[global_idx] = value;
}

int main()
{
    // -----------------------------
    // WHY: Define problem size and grid configuration.
    // 1024 elements allows us to see patterns across multiple blocks.
    // Block size 256 is a good default—multiple of 32 (warp size),
    // large enough for good occupancy, small enough to not exhaust
    // shared memory or registers.
    // -----------------------------
    const int N = 1024;
    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    // -----------------------------
    // Host (CPU) memory allocation
    // WHY: Allocate space on CPU to prepare input data and receive results.
    // -----------------------------
    int *h_in  = new int[N];
    int *h_out = new int[N];

    // -----------------------------
    // Initialize input data
    // WHY: Create predictable input pattern so we can verify correctness.
    // Simple sequential values make it easy to spot errors in output.
    // -----------------------------
    for (int i = 0; i < N; i++) {
        h_in[i] = i;
    }

    // -----------------------------
    // Device (GPU) memory allocation
    // WHY: GPU cannot access CPU memory directly. We must allocate
    // separate GPU memory (in VRAM) and explicitly copy data between
    // CPU DRAM and GPU VRAM. The CPU and GPU have completely separate
    // memory systems with different physical memory chips, different
    // memory controllers, connected by PCIe bus (slow) or NVLink (faster).
    // -----------------------------
    int *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(int));  // WHY: Allocate input buffer on GPU
    cudaMalloc(&d_out, N * sizeof(int));  // WHY: Allocate output buffer on GPU

    // -----------------------------
    // Copy input data to GPU
    // WHY: Transfer data from CPU (h_in) to GPU (d_in). This crosses the
    // PCIe bus (typically ~16 GB/s bandwidth, much slower than GPU memory's
    // ~900 GB/s). For small data, transfer time dominates compute time—
    // this is why GPUs are only worth it for large problems where compute
    // time >> transfer time. This is a SYNCHRONOUS operation—CPU waits
    // until transfer completes.
    // -----------------------------
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // -----------------------------
    // Launch kernel: memory hierarchy demo
    // WHY: Execute kernel with specified grid geometry. The GPU scheduler
    // distributes blocks across SMs, groups threads into warps, and
    // begins execution. This is ASYNCHRONOUS—the CPU continues immediately
    // without waiting for the kernel to finish (unless we explicitly sync).
    // -----------------------------
    memory_hierarchy_demo<<<blocks, threads_per_block>>>(d_in, d_out, N);

    // -----------------------------
    // Copy results back to CPU
    // WHY: GPU has computed results in d_out (GPU memory). We need them
    // in h_out (CPU memory) to inspect or use them. Another PCIe transfer.
    // cudaMemcpy implicitly synchronizes—waits for kernel to complete
    // before starting the copy, so we don't need explicit cudaDeviceSynchronize.
    // -----------------------------
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // -----------------------------
    // Print sample results
    // WHY: Verify correctness by inspecting output. Print first block only
    // (256 elements) to avoid cluttering console. In production, you'd
    // verify ALL elements programmatically or use a debugger.
    // -----------------------------
    printf("=== Memory Hierarchy Demo Results (First Block) ===\n");
    for (int i = 0; i < threads_per_block && i < N; i++) {
        printf("Index %3d -> %d\n", i, h_out[i]);
    }

    // -----------------------------
    // Launch uncoalesced demo
    // WHY: Show the performance difference between coalesced and uncoalesced
    // access. In a real application, you'd time these kernels to quantify
    // the difference. Uncoalesced access typically 2-10x slower depending
    // on the stride and cache behavior.
    // -----------------------------
    printf("\n=== Launching Uncoalesced Access Demo ===\n");
    uncoalesced_access_demo<<<blocks, threads_per_block>>>(d_out, N);
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("First few uncoalesced results:\n");
    for (int i = 0; i < 16; i++) {
        printf("Index %3d -> %d\n", i, h_out[i]);
    }

    // -----------------------------
    // Launch bank conflict demo
    // WHY: Demonstrate how shared memory bank conflicts impact performance.
    // Like uncoalesced access, you'd time this to measure the slowdown.
    // Bank conflicts can cause 2-32x slowdown depending on the conflict
    // pattern (how many threads hit the same bank).
    // -----------------------------
    printf("\n=== Launching Bank Conflict Demo ===\n");
    bank_conflict_demo<<<blocks, threads_per_block>>>(d_out, N);
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("First few bank conflict results:\n");
    for (int i = 0; i < 16; i++) {
        printf("Index %3d -> %d\n", i, h_out[i]);
    }

    // -----------------------------
    // Cleanup: free GPU memory
    // WHY: Prevent memory leaks. GPU memory is a LIMITED resource (GBs).
    // Unlike CPU, the OS doesn't automatically reclaim GPU memory when
    // your program exits—you MUST explicitly free it. Forgetting this
    // in a long-running application will eventually exhaust GPU memory.
    // -----------------------------
    cudaFree(d_in);
    cudaFree(d_out);

    // -----------------------------
    // Cleanup: free CPU memory
    // WHY: Standard C++ memory management. Delete heap-allocated arrays.
    // -----------------------------
    delete[] h_in;
    delete[] h_out;

    printf("\n=== All demos completed successfully ===\n");

    return 0;
}