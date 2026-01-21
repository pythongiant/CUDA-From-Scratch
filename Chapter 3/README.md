# Chapter 3: GPU Memory - Why It's Everything for Performance

In this chapter, you'll learn why memory is the #1 factor in GPU performance. Bad memory usage can make your code 10x slower. Good memory usage can make it 100x faster than CPU.

## The Memory Hierarchy You Must Understand

GPUs have multiple types of memory, each with different speed/cost tradeoffs:

### Global Memory (Slow but Big)
- **Size**: 8-80 GB on modern GPUs
- **Speed**: ~400-800 cycles latency (very slow!)
- **Who can access**: All threads in all blocks
- **Use for**: Input data, final results

### Shared Memory (Fast but Small)
- **Size**: 48-164 KB per processor
- **Speed**: ~20-30 cycles (100x faster than global!)
- **Who can access**: Only threads in the same block
- **Use for**: Temporary data that multiple threads need

### Registers (Fastest but Limited)
- **Size**: ~32-64 KB per processor total
- **Speed**: 1 cycle (instant!)
- **Who can access**: Only the thread that owns it
- **Use for**: Variables used within a thread

## The #1 Rule: Memory Access Patterns Matter

The way threads access memory determines if your code is fast or slow.

### Good Pattern: Coalesced Access
```cpp
// Thread 0 reads data[0], Thread 1 reads data[1], etc.
// Hardware combines 32 reads into 1 big memory transaction
int value = global_data[threadIdx.x];
```

**Result**: Full memory bandwidth utilization.

### Bad Pattern: Strided Access
```cpp
// Thread 0 reads data[0], Thread 1 reads data[2], Thread 2 reads data[4]
// Hardware can't combine - needs multiple transactions
int value = global_data[threadIdx.x * 2];
```

**Result**: 2x-10x slower!

### Worst Pattern: Random Access
```cpp
// Each thread reads a random location
int value = global_data[random_indices[threadIdx.x]];
```

**Result**: Potentially 32x slower!

## Shared Memory: The Secret Weapon

Shared memory is fast on-chip memory that threads in the same block can share.

### Use Case 1: Data Reuse
Instead of each thread loading the same data from slow global memory:

```cpp
__shared__ float shared_data[256];

// Each thread loads one value (coalesced)
shared_data[threadIdx.x] = global_data[blockIdx.x * 256 + threadIdx.x];
__syncthreads();  // Wait for all loads to complete

// Now all threads can read this data quickly
float sum = 0;
for (int i = 0; i < 256; i++) {
    sum += shared_data[i];  // Fast shared memory reads
}
```

### Use Case 2: Fixing Bad Access Patterns
When you need to transpose or reorganize data:

```cpp
__shared__ float tile[32][32];

// Load data in coalesced way
tile[threadIdx.y][threadIdx.x] = input[row * N + col];
__syncthreads();

// Write data in different pattern (also coalesced)
output[col * N + row] = tile[threadIdx.x][threadIdx.y];
```

## Bank Conflicts in Shared Memory

Shared memory is divided into 32 banks. If multiple threads access the same bank simultaneously, accesses happen one at a time (serialized).

### No Conflict (Good)
```cpp
__shared__ float data[128];
float val = data[threadIdx.x];  // Each thread hits different bank
```

### Bank Conflict (Bad)
```cpp
float val = data[threadIdx.x * 2];  // Threads hit same banks
```

## Hands-On: Run the Memory Demos

The code in this chapter demonstrates these concepts:

### 1. Memory Hierarchy Demo
Shows different memory types and their performance.

### 2. Coalesced vs Uncoalesced Access
Compare the performance difference.

### 3. Bank Conflicts
See how shared memory access patterns affect speed.

## Compile and Run

```bash
nvcc memory_management.cu -o memory_demo
./memory_demo
```

## Key Experiments to Try

### 1. Change Access Patterns
Modify the coalesced access to be uncoalesced:
```cpp
// Change this line in the kernel:
int value = global_in[global_idx];  // Coalesced
// To this:
int value = global_in[global_idx * 2];  // Uncoalesced
```
Measure the performance difference!

### 2. Modify Shared Memory Usage
Add more data reuse in the reduction example:
```cpp
// Instead of summing all elements once, sum them multiple times
for (int repeat = 0; repeat < 10; repeat++) {
    float sum = 0;
    for (int i = 0; i < blockDim.x; i++) {
        sum += shared_data[i];
    }
    // Use the sum somehow
}
```

### 3. Experiment with Bank Conflicts
Change the bank conflict pattern:
```cpp
// Try different strides
int conflict_idx = (local_idx * 4) % blockDim.x;  // Stride-4
```

## Understanding Performance

### Memory-Bound vs Compute-Bound
- **Memory-bound**: Limited by how fast you can move data (most GPU code)
- **Compute-bound**: Limited by how fast you can do calculations

### Occupancy Matters
More active threads = more warps = better at hiding memory latency.

### The Bandwidth Goal
Modern GPUs have 500-2000 GB/s memory bandwidth. Your goal: achieve 80%+ of that.

## Real-World Impact

In the matrix multiplication example:
- Bad memory access: 50 GB/s (25% of peak)
- Good memory access: 800 GB/s (80% of peak)

**4x speedup just from better memory usage!**

## Next Steps

You now understand GPU memory. In Chapter 4, we'll apply these patterns to real algorithms like reductions and image processing.

## Key Takeaways

- **Memory access patterns determine performance**
- **Coalesce global memory accesses**
- **Use shared memory for data reuse**
- **Avoid bank conflicts in shared memory**
- **Most GPU code is memory-bound, not compute-bound**
- **Good memory usage can give 10x+ speedups**