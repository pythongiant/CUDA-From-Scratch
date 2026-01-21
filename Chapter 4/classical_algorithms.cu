#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>

// ============================================================================
// PATTERN 1: MAP (Element-wise Operations)
// ============================================================================

// -----------------------------
// WHY: Map pattern - each thread operates on ONE element independently.
// No dependencies between elements, perfect for parallelization.
// This is the most common GPU pattern: transform each element of an array.
// Examples: vector add, scale, apply function element-wise.
// -----------------------------
__global__ void map_pattern(float *input, float *output, float scale, float offset, int N)
{
    // -----------------------------
    // WHY: Compute which element this thread is responsible for.
    // With N elements and thousands of threads, each thread handles
    // exactly one element. This replaces the for-loop from CPU code.
    // -----------------------------
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // -----------------------------
    // WHY: Guard against over-launch. If N isn't divisible by block size,
    // we launch extra threads that should do nothing.
    // -----------------------------
    if (idx >= N) return;

    // -----------------------------
    // WHY: The actual computation - completely independent per thread.
    // No thread reads or writes data that another thread needs.
    // Memory access is COALESCED: consecutive threads access consecutive
    // addresses, allowing the hardware to combine 32 reads into one
    // memory transaction. This is a MEMORY-BOUND kernel - limited by
    // bandwidth, not compute. Performance depends on coalescing.
    // -----------------------------
    output[idx] = input[idx] * scale + offset;
}

// ============================================================================
// PATTERN 2: REDUCE (Aggregation)
// ============================================================================

// -----------------------------
// WHY: Reduce pattern - combine N elements into one value (sum, max, min, etc).
// Cannot be done with simple parallel for-loop because each step depends on
// previous results. Solution: TREE REDUCTION using shared memory.
// This kernel does BLOCK-LEVEL reduction. Each block produces one partial sum.
// Steps: O(log N) instead of O(N), but requires log N synchronization barriers.
// -----------------------------
__global__ void reduce_sum_block(float *input, float *output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // -----------------------------
    // WHY: Shared memory for block-local reduction. Each block needs to
    // accumulate its partial sum. Using global memory would be too slow -
    // we need many reads/writes per thread. Shared memory is ~100x faster.
    // We allocate blockDim.x elements because each thread starts with one value.
    // -----------------------------
    __shared__ float shared_data[256];  // Assume max block size 256

    // -----------------------------
    // WHY: Load one element per thread from global memory (COALESCED).
    // If this thread's index exceeds N, load 0 (neutral element for sum).
    // This handles the case where N isn't a multiple of block size.
    // We load into shared memory because we'll read it multiple times
    // during the reduction tree - global memory would be too slow.
    // -----------------------------
    shared_data[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();  // WHY: Ensure ALL threads have loaded before reduction starts

    // -----------------------------
    // WHY: Tree reduction in shared memory. Each iteration, active threads
    // add pairs of elements. stride starts at blockDim.x/2 and halves each time.
    // 
    // Example with 8 threads:
    // Initial: [1, 2, 3, 4, 5, 6, 7, 8]
    // stride=4: threads 0-3 active, add elements 4 apart
    //           [1+5, 2+6, 3+7, 4+8, 5, 6, 7, 8] = [6, 8, 10, 12, ...]
    // stride=2: threads 0-1 active
    //           [6+10, 8+12, ...] = [16, 20, ...]
    // stride=1: thread 0 only
    //           [16+20, ...] = [36, ...]
    //
    // WHY stride instead of simple halving: Avoids bank conflicts in shared memory.
    // Adjacent threads access elements 'stride' apart, distributing across banks.
    // Each step needs __syncthreads() to ensure previous step completed.
    // Number of active threads HALVES each iteration - this is warp divergence,
    // but acceptable because we're doing O(log N) steps instead of O(N).
    // -----------------------------
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();  // WHY: Wait for all additions at this level to complete
    }

    // -----------------------------
    // WHY: Thread 0 writes the block's partial sum to global memory.
    // Each block produces ONE value. If we launched B blocks, we get B
    // partial sums. These need to be reduced further (either on CPU or
    // by launching another reduction kernel). This is HIERARCHICAL reduction.
    // Only thread 0 writes to avoid race conditions - shared_data[0] now
    // contains the sum of all elements in this block.
    // -----------------------------
    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

// ============================================================================
// PATTERN 3: SCAN (Prefix Sum)
// ============================================================================

// -----------------------------
// WHY: Scan (prefix sum) pattern - each output is the sum of all previous inputs.
// This looks inherently sequential, but there's a parallel algorithm.
// We implement Hillis-Steele scan: work-inefficient (O(N log N)) but simple.
// Each thread computes one output element by repeatedly looking back at
// increasing distances (1, 2, 4, 8, ...). Requires log₂N steps.
// Production code would use Blelloch scan (work-efficient O(N)) but it's complex.
// -----------------------------
__global__ void scan_hillis_steele(float *input, float *output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // -----------------------------
    // WHY: Two shared memory buffers for ping-pong. We alternate between
    // reading from one and writing to the other each iteration. This avoids
    // race conditions where we read data that another thread is simultaneously
    // writing. After each step, we swap roles: read buffer becomes write buffer.
    // -----------------------------
    __shared__ float temp[2][256];  // Double buffer

    // -----------------------------
    // WHY: Load input into first buffer. Initialize with 0 if out of bounds.
    // We use two buffers (0 and 1) to alternate between for each scan step.
    // -----------------------------
    temp[0][tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // -----------------------------
    // WHY: Hillis-Steele scan algorithm. Each iteration, look back 'offset'
    // positions and add that value to current value.
    //
    // Example with 8 elements [1,2,3,4,5,6,7,8]:
    // offset=1: [1, 1+2, 2+3, 3+4, ...] = [1, 3, 5, 7, 9, 11, 13, 15]
    // offset=2: [1, 3, 1+5, 3+7, ...] = [1, 3, 6, 10, 14, 18, 22, 26]
    // offset=4: [1, 3, 6, 10, 1+14, ...] = [1, 3, 6, 10, 15, 21, 28, 36]
    //
    // After log₂N steps, each element has accumulated all previous elements.
    // WHY ping-pong buffers: 'in_buf' reads from previous iteration's results,
    // 'out_buf' writes new results. Prevents reading partially-updated data.
    // WHY double the offset each time: This is the key insight - by doubling
    // the look-back distance, we propagate sums across the array exponentially
    // fast. After k steps, elements are summing across 2^k positions.
    // -----------------------------
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int in_buf = (offset - 1) % 2;  // Which buffer to read from
        int out_buf = offset % 2;        // Which buffer to write to

        if (tid >= offset) {
            temp[out_buf][tid] = temp[in_buf][tid] + temp[in_buf][tid - offset];
        } else {
            temp[out_buf][tid] = temp[in_buf][tid];  // No previous element to add
        }
        __syncthreads();  // WHY: All threads must finish this step before next
    }

    // -----------------------------
    // WHY: Write final result to global memory. The last buffer we wrote to
    // contains the final scan result. We need to figure out which buffer that is
    // based on how many iterations we did (log₂blockDim.x).
    // This is INCLUSIVE scan: output[i] includes input[i].
    // For EXCLUSIVE scan, we'd shift results right and set output[0] = 0.
    // NOTE: This is BLOCK-LOCAL scan only. For arrays larger than block size,
    // we'd need to: (1) scan each block, (2) scan the block sums, (3) add
    // block sums back to each block's elements. That's hierarchical scan.
    // -----------------------------
    int final_buf = 0;
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        final_buf = offset % 2;
    }

    if (idx < N) {
        output[idx] = temp[final_buf][tid];
    }
}

// ============================================================================
// PATTERN 4: STENCIL (Neighborhood Operations)
// ============================================================================

// -----------------------------
// WHY: Stencil pattern - each output depends on a NEIGHBORHOOD of inputs.
// Common in: image processing (blur, convolution), numerical methods
// (finite differences), physics simulations. Key optimization: use shared
// memory to load data once, reuse many times. Without shared memory, each
// thread would independently load its neighbors from slow global memory,
// causing massive redundant loads (3x for 3-point stencil, 9x for 3x3, etc).
// -----------------------------
__global__ void stencil_1d_3point(float *input, float *output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // -----------------------------
    // WHY: Shared memory buffer LARGER than block size. We need blockDim.x
    // elements for the block, PLUS 2 extra elements (1 on each side) for
    // the HALO REGION. Threads at block boundaries need elements from
    // adjacent blocks. Example: thread 0 needs input[idx-1], which might
    // be in the previous block. We load these "halo" elements into shared
    // memory so all threads can access their neighbors locally.
    // Size: blockDim.x + 2 = [left_halo][block_data][right_halo]
    // -----------------------------
    __shared__ float shared_data[258];  // 256 + 2 for halo

    // -----------------------------
    // WHY: Load main data - each thread loads one element. We offset by 1
    // in shared memory to leave room for the left halo element.
    // shared_data[0] is reserved for the left halo.
    // shared_data[1..blockDim.x] holds the block's actual data.
    // shared_data[blockDim.x+1] is reserved for the right halo.
    // This is COALESCED global memory access - consecutive threads load
    // consecutive addresses.
    // -----------------------------
    if (idx < N) {
        shared_data[tid + 1] = input[idx];
    } else {
        shared_data[tid + 1] = 0.0f;  // Out of bounds - use 0
    }

    // -----------------------------
    // WHY: Load HALO elements. Thread 0 loads the left neighbor (from previous
    // block), last thread loads the right neighbor (from next block).
    // These are BOUNDARY conditions - we need to handle edges carefully:
    // - If we're at array start (idx == 0), there's no left neighbor → use 0
    // - If we're at array end (idx == N-1), there's no right neighbor → use 0
    // WHY only 2 threads load halos: Loading halos is extra work. We minimize
    // it by having only edge threads do it. The alternative (all threads check
    // if they need halos) would cause massive warp divergence.
    // -----------------------------
    if (tid == 0) {
        shared_data[0] = (idx > 0) ? input[idx - 1] : 0.0f;
    }
    if (tid == blockDim.x - 1) {
        shared_data[tid + 2] = (idx < N - 1) ? input[idx + 1] : 0.0f;
    }

    __syncthreads();  // WHY: ALL threads must finish loading before we compute

    // -----------------------------
    // WHY: Compute stencil - 3-point average using shared memory.
    // Each thread reads 3 elements from shared memory (fast):
    // - shared_data[tid] = left neighbor
    // - shared_data[tid+1] = current element  
    // - shared_data[tid+2] = right neighbor
    //
    // Without shared memory, this would be 3 global memory reads per thread.
    // For a block of 256 threads, that's 768 global reads. But many are
    // REDUNDANT: thread i reads element[i-1], thread i+1 also reads element[i-1]
    // (as its left neighbor). With shared memory: we do 258 global reads
    // (256 main + 2 halo), then 768 shared memory reads (fast). We've reduced
    // global memory traffic by ~3x, which directly translates to speedup since
    // this kernel is MEMORY-BOUND.
    // -----------------------------
    if (idx < N) {
        float left = shared_data[tid];
        float center = shared_data[tid + 1];
        float right = shared_data[tid + 2];
        
        output[idx] = (left + center + right) / 3.0f;
    }
}

// ============================================================================
// HELPER: CPU REFERENCE IMPLEMENTATIONS
// ============================================================================

// -----------------------------
// WHY: CPU reference for map pattern. We'll compare GPU results against
// these to verify correctness. Simple sequential for-loop.
// -----------------------------
void cpu_map(float *input, float *output, float scale, float offset, int N)
{
    for (int i = 0; i < N; i++) {
        output[i] = input[i] * scale + offset;
    }
}

// -----------------------------
// WHY: CPU reference for reduction. Sequential accumulation.
// -----------------------------
float cpu_reduce(float *input, int N)
{
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += input[i];
    }
    return sum;
}

// -----------------------------
// WHY: CPU reference for scan. Each element accumulates all previous.
// -----------------------------
void cpu_scan(float *input, float *output, int N)
{
    output[0] = input[0];
    for (int i = 1; i < N; i++) {
        output[i] = output[i-1] + input[i];
    }
}

// -----------------------------
// WHY: CPU reference for stencil. 3-point average with boundary handling.
// -----------------------------
void cpu_stencil(float *input, float *output, int N)
{
    for (int i = 0; i < N; i++) {
        float left = (i > 0) ? input[i-1] : 0.0f;
        float center = input[i];
        float right = (i < N-1) ? input[i+1] : 0.0f;
        output[i] = (left + center + right) / 3.0f;
    }
}

// ============================================================================
// MAIN: DEMONSTRATE ALL PATTERNS
// ============================================================================

int main()
{
    // -----------------------------
    // WHY: Problem size and configuration. 1024 elements is large enough
    // to see parallel benefits but small enough to verify results manually.
    // Block size 256 is standard - multiple of 32, good occupancy.
    // -----------------------------
    const int N = 1024;
    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    // -----------------------------
    // WHY: Allocate host memory for all patterns' inputs and outputs.
    // We'll run each pattern, copy results back, and verify against CPU.
    // -----------------------------
    float *h_input = new float[N];
    float *h_output_gpu = new float[N];
    float *h_output_cpu = new float[N];

    // -----------------------------
    // WHY: Initialize input with simple pattern for easy verification.
    // Sequential values 0, 1, 2, ... make it easy to spot errors.
    // -----------------------------
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i);
    }

    // -----------------------------
    // WHY: Allocate GPU memory for input and output buffers.
    // -----------------------------
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // -----------------------------
    // WHY: Copy input data to GPU once - we'll reuse it for all patterns.
    // -----------------------------
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    printf("==========================================\n");
    printf("PATTERN 1: MAP (Element-wise Transform)\n");
    printf("==========================================\n");

    // -----------------------------
    // WHY: Test map pattern with scale=2.0, offset=1.0.
    // Each element x becomes 2*x + 1.
    // -----------------------------
    float scale = 2.0f;
    float offset = 1.0f;

    map_pattern<<<blocks, threads_per_block>>>(d_input, d_output, scale, offset, N);
    cudaMemcpy(h_output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // -----------------------------
    // WHY: Verify against CPU reference implementation. Check first 10 elements.
    // -----------------------------
    cpu_map(h_input, h_output_cpu, scale, offset, N);
    
    printf("First 10 results (GPU vs CPU):\n");
    bool map_correct = true;
    for (int i = 0; i < 10; i++) {
        printf("  [%d] GPU: %.1f, CPU: %.1f\n", i, h_output_gpu[i], h_output_cpu[i]);
        if (fabs(h_output_gpu[i] - h_output_cpu[i]) > 1e-5) {
            map_correct = false;
        }
    }
    printf("Map pattern: %s\n\n", map_correct ? "PASSED" : "FAILED");

    printf("==========================================\n");
    printf("PATTERN 2: REDUCE (Sum)\n");
    printf("==========================================\n");

    // -----------------------------
    // WHY: Test reduction. We need TWO-LEVEL reduction:
    // Level 1: Each block reduces to one value (blocks partial sums)
    // Level 2: Reduce the partial sums to final sum (on CPU for simplicity)
    // Production code would launch a second reduction kernel.
    // -----------------------------
    float *d_partial_sums;
    float *h_partial_sums = new float[blocks];
    cudaMalloc(&d_partial_sums, blocks * sizeof(float));

    reduce_sum_block<<<blocks, threads_per_block>>>(d_input, d_partial_sums, N);
    cudaMemcpy(h_partial_sums, d_partial_sums, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // -----------------------------
    // WHY: Reduce partial sums on CPU. In production, we'd launch another
    // kernel if blocks is large. Here blocks=4, so CPU is fine.
    // -----------------------------
    float gpu_sum = 0.0f;
    for (int i = 0; i < blocks; i++) {
        gpu_sum += h_partial_sums[i];
    }

    float cpu_sum = cpu_reduce(h_input, N);

    printf("GPU sum: %.1f\n", gpu_sum);
    printf("CPU sum: %.1f\n", cpu_sum);
    printf("Reduce pattern: %s\n\n", 
           (fabs(gpu_sum - cpu_sum) < 1e-3) ? "PASSED" : "FAILED");

    printf("==========================================\n");
    printf("PATTERN 3: SCAN (Prefix Sum)\n");
    printf("==========================================\n");

    // -----------------------------
    // WHY: Test scan pattern. This is block-local scan only.
    // For multi-block scan, we'd need hierarchical scan (scan blocks,
    // scan block sums, add block sums back). We only test one block here.
    // -----------------------------
    const int scan_N = 256;  // One block only for simplicity
    
    scan_hillis_steele<<<1, scan_N>>>(d_input, d_output, scan_N);
    cudaMemcpy(h_output_gpu, d_output, scan_N * sizeof(float), cudaMemcpyDeviceToHost);

    cpu_scan(h_input, h_output_cpu, scan_N);

    printf("First 10 scan results (GPU vs CPU):\n");
    bool scan_correct = true;
    for (int i = 0; i < 10; i++) {
        printf("  [%d] GPU: %.1f, CPU: %.1f\n", i, h_output_gpu[i], h_output_cpu[i]);
        if (fabs(h_output_gpu[i] - h_output_cpu[i]) > 1e-3) {
            scan_correct = false;
        }
    }
    printf("Scan pattern: %s\n\n", scan_correct ? "PASSED" : "FAILED");

    printf("==========================================\n");
    printf("PATTERN 4: STENCIL (3-point average)\n");
    printf("==========================================\n");

    // -----------------------------
    // WHY: Test stencil pattern. Each output is average of 3 neighbors.
    // -----------------------------
    stencil_1d_3point<<<blocks, threads_per_block>>>(d_input, d_output, N);
    cudaMemcpy(h_output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    cpu_stencil(h_input, h_output_cpu, N);

    printf("First 10 stencil results (GPU vs CPU):\n");
    bool stencil_correct = true;
    for (int i = 0; i < 10; i++) {
        printf("  [%d] GPU: %.2f, CPU: %.2f\n", i, h_output_gpu[i], h_output_cpu[i]);
        if (fabs(h_output_gpu[i] - h_output_cpu[i]) > 1e-3) {
            stencil_correct = false;
        }
    }
    printf("Stencil pattern: %s\n\n", stencil_correct ? "PASSED" : "FAILED");

    // -----------------------------
    // WHY: Cleanup - free all allocated memory (GPU and CPU).
    // -----------------------------
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_partial_sums);
    
    delete[] h_input;
    delete[] h_output_gpu;
    delete[] h_output_cpu;
    delete[] h_partial_sums;

    printf("==========================================\n");
    printf("All pattern demonstrations completed!\n");
    printf("==========================================\n");

    return 0;
}