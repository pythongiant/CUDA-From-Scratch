# Chapter 4: Common Parallel Patterns - Speed Up Real Algorithms

Now you know the basics! This chapter shows you **ready-to-use patterns** for speeding up real computations. These patterns appear in most GPU-accelerated code.

## What You'll Learn

Four fundamental patterns that solve 90% of parallel problems:
1. **Map** - Apply the same operation to each element
2. **Reduce** - Combine elements into a single value
3. **Scan** - Compute running totals
4. **Stencil** - Process neighborhoods (like image filters)

## Pattern 1: Map - Element-wise Operations

**Use when**: Each output depends only on the corresponding input.

**Examples**: Adding vectors, scaling arrays, applying math functions.

### Simple Map Example
```cpp
__global__ void map_pattern(float *input, float *output, float scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    output[idx] = input[idx] * scale;  // Each thread: one operation
}
```

**Why it works**: No dependencies between elements. Thread 5's work doesn't affect thread 10's work.

**Performance**: Memory-bound (limited by how fast you can read/write data).

## Pattern 2: Reduce - Summing/Combining Elements

**Use when**: You need to combine all elements into one value (sum, max, min, etc.).

**Challenge**: Sequential sum is slow. Parallel tree reduction is fast!

### Tree Reduction Example
```cpp
__global__ void reduce_sum(float *input, float *block_sums, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ float shared_data[256];

    // Load data
    shared_data[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // Tree reduction: halve active threads each step
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes this block's sum
    if (tid == 0) block_sums[blockIdx.x] = shared_data[0];
}
```

**Why it works**: Instead of N sequential steps, do log₂N parallel steps.

**Speedup**: From O(N) to O(log N) time!

## Pattern 3: Scan - Running Totals

**Use when**: Each output needs the sum/total of all previous elements.

**Examples**: Cumulative sums, finding positions in sorted data.

### Parallel Scan Example (Hillis-Steele)
```cpp
__global__ void scan_hillis_steele(float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ float temp[256];

    temp[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // Each step: add from increasing distances
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        float val = 0.0f;
        if (tid >= offset) {
            val = temp[tid - offset];
        }
        __syncthreads();
        if (tid >= offset) {
            temp[tid] += val;
        }
        __syncthreads();
    }

    if (idx < N) output[idx] = temp[tid];
}
```

**Input**: [1, 2, 3, 4]
**Output**: [1, 3, 6, 10] (cumulative sums)

## Pattern 4: Stencil - Neighborhood Operations

**Use when**: Each output depends on nearby inputs.

**Examples**: Image blurring, physics simulations, edge detection.

### 1D Stencil Example (3-point average)
```cpp
__global__ void stencil_1d(float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ float shared[258];  // Block data + halo

    // Load main data
    if (idx < N) {
        shared[tid + 1] = input[idx];
    } else {
        shared[tid + 1] = 0.0f;
    }

    // Load halo (boundary elements)
    if (tid == 0) {
        shared[0] = (idx > 0) ? input[idx - 1] : 0.0f;
    }
    if (tid == blockDim.x - 1) {
        shared[tid + 2] = (idx < N - 1) ? input[idx + 1] : 0.0f;
    }
    __syncthreads();

    // Compute average of 3 neighbors
    if (idx < N) {
        float left = shared[tid];
        float center = shared[tid + 1];
        float right = shared[tid + 2];
        output[idx] = (left + center + right) / 3.0f;
    }
}
```

**Why shared memory?**: Instead of 768 global memory loads, do 258 loads + fast shared reads.

## Hands-On: Run All Patterns

The code demonstrates all four patterns with working examples:

```bash
nvcc classical_algorithms.cu -o patterns_demo
./patterns_demo
```

You'll see:
- Map: Element-wise scaling
- Reduce: Sum of 1024 elements
- Scan: Cumulative sums
- Stencil: 3-point averaging

## Key Experiments

### 1. Modify the Map Operation
```cpp
// Try different operations
output[idx] = sin(input[idx]) + cos(input[idx]);
output[idx] = (input[idx] > 0.5f) ? 1.0f : 0.0f;  // Threshold
```

### 2. Change Reduction Operation
```cpp
// Instead of sum, find maximum
if (tid < stride) {
    shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
}
```

### 3. Modify Stencil Pattern
```cpp
// 5-point stencil instead of 3-point
// Need bigger halo region!
```

## When NOT to Use These Patterns

- **Small data**: CPU might be faster due to GPU launch overhead
- **Sequential dependencies**: If each step depends on the previous result
- **Random memory access**: Breaks coalescing, kills performance

## Real-World Applications

- **Map**: Image processing, neural network forward pass
- **Reduce**: Computing loss in ML, finding best matches
- **Scan**: Sorting algorithms, parallel prefix sums
- **Stencil**: Game physics, weather simulation, computer vision

## Next Steps

You now have tools for most parallel computations! In Chapter 5, we'll connect this to real Python/PyTorch code.

## Key Takeaways

- **Map**: Independent element operations
- **Reduce**: Tree-based aggregation (fast!)
- **Scan**: Parallel prefix computations
- **Stencil**: Neighborhood processing with shared memory
- **These patterns solve 90% of parallel problems**
- **Always analyze dependencies first**
