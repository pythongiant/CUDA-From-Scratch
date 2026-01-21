# Chapter 2: Your First CUDA Program - Understanding How Code Runs on GPU

Now we get hands-on! You'll write and run your first CUDA program. We'll see how the abstract concepts from Chapter 1 become real code.

## What You'll Build

A program that demonstrates:
- Writing a function that runs on the GPU (kernel)
- Launching 256 copies of that function simultaneously
- Each "copy" processes different data
- Seeing the results back on the CPU

## The Basic Structure

Every CUDA program has two parts:

1. **CPU code** - main program, memory management
2. **GPU code** - the parallel kernel function

## Your First Kernel

Here's the GPU function we'll write:

```cpp
__global__ void execution_model_demo(int *out, int N)
{
    // Each thread figures out which data element it should handle
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx  = threadIdx.x;

    // Skip if this thread is beyond our data size
    if (global_idx >= N) return;

    // Calculate which "warp" this thread is in (group of 32 threads)
    int warp_id   = local_idx / 32;
    int lane_id   = local_idx % 32;

    // Intentional branching to show warp divergence
    int value;
    if (lane_id < 16) {
        value = global_idx * 2;  // First half of warp does this
    } else {
        value = global_idx * 3;  // Second half does this
    }

    // All threads in block must reach here before any can continue
    __syncthreads();

    // Write result to memory
    out[global_idx] = value;
}
```

## Understanding the Key Parts

### Thread Identity
```cpp
int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
```

- `threadIdx.x` - This thread's position within its block (0-63)
- `blockIdx.x` - Which block this thread is in (0-3 for our example)
- `blockDim.x` - How many threads per block (64 in our case)

So thread 5 in block 2 has `global_idx = 2 * 64 + 5 = 133`.

### Warps - The Real Execution Unit
```cpp
int warp_id = local_idx / 32;    // Which group of 32 threads
int lane_id = local_idx % 32;    // Position within the group
```

GPU hardware runs threads in groups of 32 called **warps**. All 32 threads in a warp execute the same instruction at the same time.

### Warp Divergence - Why Branching Hurts
```cpp
if (lane_id < 16) {
    value = global_idx * 2;
} else {
    value = global_idx * 3;
}
```

Within one warp:
- Threads 0-15: multiply by 2
- Threads 16-31: multiply by 3

The hardware runs these **sequentially**, not in parallel! The whole warp waits while first the "if" executes, then the "else".

### Synchronization
```cpp
__syncthreads();
```

All threads in the same block must reach this point before any can continue. This is crucial when threads need to share data.

## Running the Program

### 1. Compile
```bash
nvcc execute_model_demo.cu -o demo
```

### 2. Run
```bash
./demo
```

### 3. What You'll See
```
Index   0 -> 0    # lane_id=0 (<16) so 0*2 = 0
Index   1 -> 2    # lane_id=1 (<16) so 1*2 = 2
...
Index  15 -> 30   # lane_id=15 (<16) so 15*2 = 30
Index  16 -> 48   # lane_id=16 (>=16) so 16*3 = 48
Index  17 -> 51   # lane_id=17 (>=16) so 17*3 = 51
...
```

Notice the pattern change at index 16, 48, 80, etc. - this shows where warps begin.

## Launch Configuration

When you launch a kernel:

```cpp
execution_model_demo<<<blocks, threads_per_block>>>(d_out, N);
```

- `blocks = 4` - Launch 4 blocks
- `threads_per_block = 64` - 64 threads each
- Total: 256 threads for N=256 elements

The GPU scheduler assigns blocks to available processors and manages execution.

## Key Concepts You Just Saw

- **Threads are independent workers** - Each handles one data element
- **Warps execute together** - 32 threads move as a unit
- **Divergence hurts performance** - Branches within a warp slow everything down
- **Blocks can synchronize** - `__syncthreads()` coordinates within a block
- **No guaranteed order** - Threads finish in any order

## Try Modifying the Code

1. **Change the branching condition:**
   ```cpp
   if (lane_id < 8) {  // Only first quarter
       value = global_idx * 2;
   } else {
       value = global_idx * 3;
   }
   ```
   See how this affects performance.

2. **Add more synchronization:**
   ```cpp
   __syncthreads();
   // Do some work
   __syncthreads();
   // Do more work
   ```

3. **Experiment with different block sizes:**
   ```cpp
   const int threads_per_block = 128;  // Try 32, 64, 128, 256
   ```

## What Happens Inside the GPU

When you launch the kernel:

1. GPU creates 4 blocks of 64 threads each = 256 threads
2. Each block gets assigned to a processor
3. Processors break threads into warps (groups of 32)
4. Warps execute instructions, switching rapidly to hide memory delays
5. When threads diverge, execution serializes within the warp
6. Results get written to GPU memory, then copied back to CPU

## Next Steps

You now understand how CUDA code actually executes! In Chapter 3, we'll learn about GPU memory - why it's so important for performance.

## Key Takeaways

- **Kernels are functions that run on GPU**
- **Threads are independent workers**
- **Warps (32 threads) execute together**
- **Branching within warps hurts performance**
- **Blocks can synchronize with `__syncthreads()`**
- **Launch configuration controls parallelism**