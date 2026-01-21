# Chapter 2 — The CUDA Execution Model (From Code to Hardware)

This chapter explains **how a CUDA kernel actually runs on real GPU hardware**, and why CUDA looks the way it does.

---

## 2.1 The Three-Level Hierarchy

CUDA's execution model maps software to hardware through three nested levels:

### Grid → Streaming Multiprocessors (SMs)
When you launch a kernel with `<<<blocks, threads>>>`, you create a **grid** of blocks. The GPU's scheduler distributes these blocks across available SMs. Each SM can run multiple blocks concurrently, but blocks never migrate—once assigned, a block stays on its SM until completion.

### Block → SM Resident Workload
A block is a collection of threads that share fast on-chip memory and can synchronize with `__syncthreads()`. All threads in a block execute on the same SM, which is why synchronization works—they share hardware resources. The block size you choose (like `threads_per_block = 64`) determines resource usage per block.

### Warp → Execution Unit
The SM doesn't execute individual threads. It executes **warps**—groups of 32 threads in lockstep. This is the fundamental execution width of NVIDIA GPUs. Every thread in a warp executes the same instruction at the same time on the SM's CUDA cores.

---

## 2.2 Why Warps Exist

The warp is hardware reality, not abstraction. Modern GPUs achieve high throughput by executing the same instruction across 32 threads simultaneously—Single Instruction, Multiple Thread (SIMT) execution.

```cpp
int warp_id = local_idx / 32;
int lane_id = local_idx % 32;
```

These calculations expose the underlying execution groups. In a block of 64 threads:
- Threads 0-31 form warp 0
- Threads 32-63 form warp 1

Each warp gets scheduled independently, but threads within a warp move together.

---

## 2.3 Warp Divergence — The Hidden Cost

Consider this branching code:

```cpp
if (lane_id < 16) {
    value = global_idx * 2;
} else {
    value = global_idx * 3;
}
```

Within a single warp, 16 threads take the first path and 16 take the second. But the hardware executes both paths **serially**:

1. First, lanes 0-15 execute `value = global_idx * 2` while lanes 16-31 wait (masked off)
2. Then, lanes 16-31 execute `value = global_idx * 3` while lanes 0-15 wait

The warp takes the time of both paths combined. This is **warp divergence**—when threads in the same warp follow different control flow paths, execution serializes. Performance degrades proportionally to the number of divergent paths.

Different warps can diverge freely without penalty—they execute independently. Divergence only matters within a warp.

---

## 2.4 Block-Level Synchronization

```cpp
__syncthreads();
```

This barrier ensures all threads in the block reach this point before any proceed. It's implemented in hardware using the SM's synchronization logic. Crucially:

- **Scope**: Only threads in the same block synchronize
- **Hardware requirement**: All threads in the block must be resident on the same SM
- **Warp consideration**: If any thread in a warp can reach the barrier, all threads in that warp must be able to reach it (otherwise deadlock)

You cannot synchronize across blocks—they may run on different SMs, at different times, or even be scheduled after other blocks complete.

---

## 2.5 Memory and Thread Identity

```cpp
int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
```

This formula maps a thread's 2D identity (block + position within block) to a 1D index for memory access. The GPU provides:

- `blockIdx`: Which block this thread belongs to
- `blockDim`: How many threads per block (constant across launch)
- `threadIdx`: Position within the block (0 to `blockDim - 1`)

Each thread uses these to compute unique memory locations:

```cpp
out[global_idx] = value;
```

Memory writes happen independently—there's no guaranteed ordering between threads unless you explicitly synchronize. The hardware may coalesce adjacent memory accesses from the same warp into efficient transactions, but that's a performance optimization, not a semantic guarantee.

---

## 2.6 The Guard Pattern

```cpp
if (global_idx >= N) return;
```

Grids usually launch more threads than needed for clean division:

```cpp
int blocks = (N + threads_per_block - 1) / threads_per_block;
```

If `N = 256` and `threads_per_block = 64`, you get 4 blocks = 256 threads exactly. But if `N = 250`, you still launch 4 blocks = 256 threads. The extra 6 threads in the last warp must check bounds and exit early.

Early returns don't help performance within a warp—if any thread continues, the whole warp waits. But they prevent out-of-bounds memory writes.

---

## 2.7 Launch Configuration

```cpp
execution_model_demo<<<blocks, threads_per_block>>>(d_out, N);
```

The `<<<...>>>` syntax specifies the grid geometry:
- **Left parameter** (`blocks`): How many blocks in the grid
- **Right parameter** (`threads_per_block`): How many threads per block

Choosing these values involves tradeoffs:

**Block size too small**: Underutilizes SM resources, exposes scheduling overhead

**Block size too large**: May limit blocks per SM due to resource constraints (registers, shared memory)

Common block sizes: 128, 256, 512 threads (always multiples of 32 for warp alignment)

---

## 2.8 What the Hardware Actually Does

When this kernel launches:

1. **Grid distribution**: The GPU scheduler assigns blocks to available SMs
2. **Warp formation**: Each SM groups block threads into warps of 32
3. **Warp scheduling**: The SM rapidly switches between warps to hide latency (memory access, ALU operations)
4. **SIMT execution**: Each warp executes one instruction across 32 threads per cycle
5. **Divergence handling**: Warps serialize execution at branches, using predication masks
6. **Synchronization**: `__syncthreads()` pauses warp scheduling until all block warps reach the barrier
7. **Memory operations**: Coalesced writes from warps to global memory

The result you see in the output reflects this execution—each thread computed its value, but the order of execution and memory writes followed hardware scheduling, not source code order.

---

## Key Takeaways

- **Warps are real**: 32 threads execute as a unit on the hardware
- **Divergence costs performance**: Branching within a warp serializes execution
- **Blocks enable cooperation**: Shared memory and synchronization work within a block
- **Memory accesses are independent**: No ordering guarantees without explicit synchronization
- **Resource limits matter**: Block size affects SM occupancy and performance

The CUDA execution model exists because this is how GPU hardware achieves massive parallelism—thousands of threads sharing hardware through rapid context switching and SIMT execution. Understanding this mapping from code to hardware is essential for writing efficient CUDA kernels.