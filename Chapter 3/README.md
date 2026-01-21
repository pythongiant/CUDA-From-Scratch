# Chapter 3 — Memory Hierarchy and Access Patterns

This chapter explains **how GPU memory actually works**, why different memory types exist, and how access patterns determine performance. The GPU's memory system is fundamentally different from CPU memory—understanding these differences is critical for writing fast kernels.

---

## 3.1 The Memory Hierarchy

GPUs have multiple memory spaces, each with different characteristics:

### Global Memory — Large but Slow
Global memory is the main DRAM attached to the GPU (typically GBs). Every thread can read/write it, and it persists across kernel launches. But it's far from the compute cores—access latency is 400-800 cycles. This is where you allocate with `cudaMalloc`.

**Characteristics:**
- Capacity: Gigabytes (8GB - 80GB+ on modern GPUs)
- Latency: ~400-800 cycles
- Bandwidth: 100s of GB/s to 1+ TB/s (depending on GPU)
- Scope: Visible to all threads across all blocks
- Lifetime: Persistent until explicitly freed

### Shared Memory — Fast but Small
Shared memory is on-chip SRAM on each SM. Only threads within the same block can access it. It's 100x faster than global memory but limited to ~48KB-164KB per SM.

**Characteristics:**
- Capacity: 48KB-164KB per SM (configurable on some GPUs)
- Latency: ~20-30 cycles (comparable to L1 cache)
- Bandwidth: Multiple TB/s (on-chip)
- Scope: Block-local only
- Lifetime: Exists only during block execution

### Registers — Fastest but Most Limited
Each thread gets private register storage. Registers are the fastest memory (1-cycle access) but extremely limited—typically 32KB-64KB worth per SM, divided among all active threads.

**Characteristics:**
- Capacity: 255 registers per thread (maximum), but limited by SM total
- Latency: 1 cycle
- Scope: Thread-private
- Lifetime: Exists only during thread execution

### L1/L2 Cache — Automatic but Unpredictable
The GPU has hardware caches that automatically cache global memory accesses. You don't control them directly, but access patterns determine hit rates.

---

## 3.2 Memory Coalescing — The Critical Pattern

The most important concept for GPU memory performance is **coalescing**. When threads in a warp access memory, the hardware tries to combine their individual requests into a single transaction.

### Perfect Coalescing
```cpp
// Thread 0 accesses data[0]
// Thread 1 accesses data[1]
// Thread 2 accesses data[2]
// ...
// Thread 31 accesses data[31]
data[threadIdx.x] = value;
```

Threads in a warp access **consecutive addresses** → hardware combines into one 128-byte transaction → full bandwidth utilization.

### Uncoalesced Access (Stride)
```cpp
// Thread 0 accesses data[0]
// Thread 1 accesses data[2]
// Thread 2 accesses data[4]
// ...
// Thread 31 accesses data[62]
data[threadIdx.x * 2] = value;
```

Threads access every other element (stride-2) → hardware issues multiple transactions → wasted bandwidth, lower performance.

### Worst Case (Random Access)
```cpp
data[random_indices[threadIdx.x]] = value;
```

Completely random addresses → potentially 32 separate transactions for one warp → 32x slower than coalesced access.

---

## 3.3 Shared Memory Use Cases

Shared memory exists to solve two problems:

### Problem 1: Data Reuse
When multiple threads need the same data from global memory, load it once into shared memory, then read it many times:

```cpp
__shared__ float tile[TILE_SIZE];

// One coalesced load from global memory
tile[threadIdx.x] = global_data[blockIdx.x * TILE_SIZE + threadIdx.x];
__syncthreads();

// Many fast reads from shared memory
for (int i = 0; i < TILE_SIZE; i++) {
    sum += tile[i];  // Reuse data loaded by other threads
}
```

### Problem 2: Access Pattern Transformation
When global memory access patterns would be uncoalesced, stage data through shared memory to reorganize it:

**Matrix Transpose Example:**
```cpp
// Bad: Uncoalesced writes (column-major)
output[col * N + row] = input[row * N + col];

// Good: Stage through shared memory
__shared__ float tile[TILE_SIZE][TILE_SIZE];
tile[threadIdx.y][threadIdx.x] = input[row * N + col];  // Coalesced read
__syncthreads();
output[col * N + row] = tile[threadIdx.x][threadIdx.y]; // Coalesced write
```

Both global memory operations are now coalesced, even though we're transposing.

---

## 3.4 Bank Conflicts in Shared Memory

Shared memory is organized into 32 **banks** (one per warp lane). If multiple threads in a warp access different addresses in the **same bank**, accesses serialize—similar to warp divergence but for memory.

### No Conflict (Different Banks)
```cpp
__shared__ float data[128];
// Each thread accesses a different bank
float val = data[threadIdx.x];  // threadIdx.x = 0-31
```

Thread 0 → bank 0, Thread 1 → bank 1, ..., Thread 31 → bank 31. All accesses happen in parallel.

### Bank Conflict (Same Bank, Different Addresses)
```cpp
__shared__ float data[128];
// Stride-2 access: threads 0 and 16 both access bank 0
float val = data[threadIdx.x * 2];
```

Banks are assigned modulo 32, so addresses 0 and 64 map to the same bank. Multiple threads accessing the same bank (but different addresses) causes a **bank conflict**—accesses serialize.

### Broadcast (Same Address, No Conflict)
```cpp
// All threads read the SAME address
float val = data[0];
```

When all threads read the same address, the hardware broadcasts it—no conflict, full speed.

---

## 3.5 Register Pressure and Occupancy

Registers are allocated per-thread. If your kernel uses too many registers, fewer threads fit on the SM simultaneously:

**Example:**
- SM has 65,536 registers total
- Your kernel uses 64 registers per thread
- Maximum threads per SM: 65,536 / 64 = 1,024 threads
- If hardware supports 2,048 threads per SM, you've lost 50% occupancy

**Consequences:**
- Lower occupancy → fewer warps to hide memory latency
- GPU stalls waiting for memory → lower throughput

**Solutions:**
- Reduce local variables (each becomes registers)
- Use compiler flag `-maxrregcount` to limit register usage
- Spill to local memory (slow but better than nothing)

---

## 3.6 Memory Access Latency Hiding

The GPU doesn't wait for memory—it switches to other warps:

**The Mechanism:**
1. Warp A issues a global memory load (400 cycles latency)
2. SM immediately switches to Warp B (no stall)
3. Warp B executes instructions while Warp A waits
4. Eventually data arrives for Warp A
5. SM switches back to Warp A when it's ready

**Key Insight:** You need **enough active warps** to hide latency. If you only have 2 warps per SM and both are waiting for memory, the SM stalls. If you have 16 warps, at least some are always ready to execute.

This is why **occupancy** matters—more active threads means more warps to switch between.

---

## 3.7 Constant and Texture Memory

### Constant Memory
Small read-only memory (~64KB) with broadcast capability. Best when all threads in a warp read the **same address**:

```cpp
__constant__ float coefficients[256];

// All threads read coefficients[5] → broadcasted, fast
float c = coefficients[5];
```

If threads read different addresses, it serializes like bank conflicts.

### Texture Memory
Specialized for 2D/3D spatial locality with hardware filtering. Cached separately from global memory. Useful for image processing where neighboring threads access neighboring pixels.

---

## 3.8 Unified Memory (Managed Memory)

Modern CUDA provides unified memory—single pointer accessible from CPU and GPU:

```cpp
float *data;
cudaMallocManaged(&data, N * sizeof(float));

// CPU can access data[i]
// GPU can access data[i]
// Runtime handles transfers automatically
```

**Benefits:** Simpler programming, no explicit copies

**Drawbacks:** Hidden transfer costs, potential page faults, harder to optimize

Useful for prototyping, but explicit memory management gives better performance for production code.

---

## 3.9 Memory Throughput Calculations

**Theoretical Maximum:**
- GPU has 900 GB/s bandwidth (example: A100)
- Kernel reads 4 bytes per thread
- Processes 1 billion elements
- Minimum time: 4 GB / 900 GB/s = 4.4 ms

**Achieved Performance:**
- Actual time: 10 ms
- Achieved bandwidth: 4 GB / 10 ms = 400 GB/s
- Efficiency: 400/900 = 44%

The gap comes from uncoalesced access, bank conflicts, insufficient occupancy, or memory-bound operations mixed with computation.

---

## 3.10 Memory-Bound vs Compute-Bound

**Memory-Bound Kernel:**
```cpp
// Simple copy: limited by memory bandwidth
output[i] = input[i];
```

Spends most time waiting for memory. Performance determined by bandwidth, not compute throughput.

**Compute-Bound Kernel:**
```cpp
// Complex math: many operations per memory access
for (int j = 0; j < 1000; j++) {
    sum += sin(input[i] * j) * cos(input[i] * j);
}
output[i] = sum;
```

Loads one value, does extensive computation. Performance limited by ALU throughput.

**Arithmetic Intensity:** Operations per byte loaded. Higher intensity → more compute-bound → better GPU utilization.

---

## Key Takeaways

- **Global memory is slow**: Minimize accesses, maximize coalescing
- **Shared memory enables cooperation**: Use it for data reuse and access pattern transformation
- **Registers are fastest but limited**: Watch register pressure to maintain occupancy
- **Coalescing determines bandwidth**: Consecutive access patterns are critical
- **Bank conflicts serialize shared memory**: Structure data to avoid same-bank conflicts
- **Occupancy hides latency**: More active warps → less waiting for memory
- **Memory hierarchy is deep**: Each level has different tradeoffs in size, speed, and scope