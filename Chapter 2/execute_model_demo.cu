#include <cstdio>
#include <cuda_runtime.h>

__global__ void execution_model_demo(int *out, int N)
{
    // -----------------------------
    // Logical thread identity
    // WHY: Every thread needs to know (1) which element of the array it owns,
    // and (2) its position within its block for synchronization purposes.
    // global_idx maps the 2D grid structure (blocks × threads) to a 1D array index.
    // local_idx is needed because __syncthreads() only works within a block,
    // and warp calculations depend on position within the block, not globally.
    // -----------------------------
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx  = threadIdx.x;

    // -----------------------------
    // Guard against over-launch
    // WHY: We launch blocks × threads_per_block total threads, but this number
    // must be a multiple of the block size. If N isn't evenly divisible by block size,
    // we over-launch threads. Without this guard, extra threads would write beyond
    // the array bounds, causing memory corruption. Early return prevents invalid writes.
    // Note: Early return doesn't save cycles within a warp—if any thread continues,
    // the whole warp continues—but it prevents undefined behavior.
    // -----------------------------
    if (global_idx >= N) return;

    // -----------------------------
    // Warp-level behavior
    // Warp size is 32 on NVIDIA GPUs
    // WHY: The SM doesn't execute individual threads—it executes warps (groups of 32).
    // We calculate warp_id and lane_id to demonstrate how threads are actually grouped
    // at the hardware level. warp_id tells you which execution unit you're in,
    // lane_id tells you your position within that unit. These aren't just abstractions—
    // they reflect how the hardware schedules and executes instructions. Understanding
    // warp membership is critical for avoiding divergence and optimizing memory access.
    // -----------------------------
    int warp_id   = local_idx / 32;
    int lane_id   = local_idx % 32;

    // -----------------------------
    // Intentional warp divergence
    // Threads in the SAME warp will follow different paths
    // WHY: This demonstrates the performance cost of branching within a warp.
    // Because lane_id < 16 splits each warp in half, the hardware must execute
    // BOTH branches serially—first the if-path for lanes 0-15 (while 16-31 wait),
    // then the else-path for lanes 16-31 (while 0-15 wait). The warp takes the
    // time of BOTH paths combined. This is why divergent branches within a warp
    // kill performance—you get serial execution on parallel hardware.
    // If different WARPS take different paths, there's no penalty—they execute
    // independently. Divergence only matters WITHIN a warp.
    // -----------------------------
    int value;
    if (lane_id < 16) {
        value = global_idx * 2;
    } else {
        value = global_idx * 3;
    }

    // -----------------------------
    // Block-level synchronization
    // Only threads in THIS block
    // WHY: This barrier ensures all threads in the block reach this point before
    // any proceed. We use it here to demonstrate its scope—it ONLY synchronizes
    // threads in the same block (they're on the same SM, sharing resources).
    // You CANNOT synchronize across blocks—they may run on different SMs, at
    // different times, or even sequentially. The hardware implements this using
    // per-SM synchronization logic. Without __syncthreads(), threads race ahead
    // independently, which matters when they share data via shared memory
    // (not shown here, but that's the primary use case).
    // -----------------------------
    __syncthreads();

    // -----------------------------
    // Write result
    // Each thread writes independently
    // No global ordering guarantees
    // WHY: Global memory writes happen asynchronously and independently per thread.
    // There's NO guaranteed order between threads without explicit synchronization.
    // Thread 100 might write before thread 5. The hardware may coalesce adjacent
    // writes from the same warp into a single memory transaction for efficiency,
    // but that's a performance optimization, not a correctness guarantee.
    // Each thread writes to a unique location (global_idx), so there's no race condition.
    // If multiple threads wrote to the same location, behavior would be undefined
    // without atomics.
    // -----------------------------
    out[global_idx] = value;
}

int main()
{
    // WHY: N defines the problem size—how many elements we're processing.
    const int N = 256;
    
    // WHY: Block size (64) is chosen to balance resource usage and occupancy.
    // Must be a multiple of 32 (warp size) to avoid wasting SM resources.
    // 64 means 2 warps per block. Too small wastes scheduling opportunities,
    // too large may limit how many blocks fit on an SM due to register/shared
    // memory limits. 64-512 is typical; 256 is most common for simple kernels.
    const int threads_per_block = 64;
    
    // WHY: This formula ensures we launch enough threads to cover all N elements.
    // (N + threads_per_block - 1) / threads_per_block is ceiling division.
    // For N=256, threads_per_block=64: (256+63)/64 = 319/64 = 4 blocks exactly.
    // For N=250: (250+63)/64 = 313/64 = 4 blocks (256 threads), hence the guard.
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    // WHY: h_out is host (CPU) memory for storing results after GPU computation.
    int *h_out = new int[N];
    
    // WHY: d_out is device (GPU) memory. The GPU cannot directly access CPU memory
    // (different physical address spaces, different memory controllers).
    int *d_out;

    // WHY: Allocate GPU memory. cudaMalloc is analogous to malloc but for GPU.
    // The GPU will write results here; later we copy them back to CPU.
    cudaMalloc(&d_out, N * sizeof(int));

    // WHY: <<<blocks, threads_per_block>>> is the launch configuration—it defines
    // the grid geometry. This creates a 1D grid of 'blocks' blocks, each containing
    // 'threads_per_block' threads. The GPU scheduler distributes these blocks across
    // available SMs. This syntax is CUDA-specific; it tells the runtime how to map
    // the parallel work onto hardware. Without this, we'd just have serial code.
    execution_model_demo<<<blocks, threads_per_block>>>(d_out, N);

    // WHY: The kernel executes on the GPU; results are in GPU memory (d_out).
    // We must explicitly copy them back to CPU memory (h_out) to inspect them.
    // The CPU and GPU have separate memory systems—they don't automatically share data.
    // This copy happens over PCIe bus (slow) or NVLink (faster), taking microseconds.
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // -----------------------------
    // Inspect results
    // WHY: Print first 64 elements (one full block) to verify correctness.
    // We print only a subset because 256 lines would clutter output.
    // This lets us see the pattern: lane_id < 16 produces value = idx*2,
    // lane_id >= 16 produces value = idx*3. We can verify the divergence
    // behavior produced the expected results despite serialized execution.
    // -----------------------------
    for (int i = 0; i < 64; i++) {
        printf("Index %3d -> %d\n", i, h_out[i]);
    }

    // WHY: Free GPU memory to prevent memory leaks. GPU memory is a limited resource
    // (typically GBs). Unlike CPU memory, the OS doesn't automatically reclaim GPU
    // memory when the process exits—you must explicitly free it.
    cudaFree(d_out);
    
    // WHY: Free CPU memory allocated with new. Standard C++ memory management.
    delete[] h_out;

    return 0;
}