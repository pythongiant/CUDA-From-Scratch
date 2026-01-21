# Chapter 4 — Mapping Classical Algorithms to CUDA

This chapter explains **how to transform sequential CPU algorithms into parallel GPU kernels**. We'll examine the fundamental patterns that appear repeatedly: map, reduce, scan, and stencil. Understanding these patterns is essential because most real algorithms are combinations of these primitives.

---

## 3.1 The Parallel Thinking Shift

CPU algorithms are designed for sequential execution—one step follows another. GPU algorithms require **decomposition**: breaking problems into independent subproblems that can execute simultaneously.

### Sequential Mindset
```
for (int i = 0; i < N; i++) {
    result[i] = compute(input[i]);
}
```
Time: O(N) — processes one element per iteration.

### Parallel Mindset
```
// Launch N threads, each processes one element
result[threadIdx.x] = compute(input[threadIdx.x]);
```
Time: O(1) — all elements process simultaneously (ignoring hardware limits).

**The Key Question:** Can I partition this problem so that multiple threads work on different parts **without interfering with each other**?

---

## 3.2 Pattern 1: Map (Element-wise Operations)

**Map** applies the same operation independently to each element. This is the simplest and most common parallel pattern.

### Characteristics
- **No dependencies** between elements
- **Perfect parallelism**: each thread operates on one element
- **Memory pattern**: typically coalesced reads and writes
- **Examples**: vector addition, scaling, element-wise functions

### CPU Version (Sequential)
```cpp
for (int i = 0; i < N; i++) {
    output[i] = input[i] * scale + offset;
}
```

### GPU Version (Parallel)
Each thread handles one element. The for-loop disappears—it's replaced by launching N threads.

**Why This Works:**
- No element depends on any other element's result
- No shared state between threads
- Order of execution doesn't matter
- Natural mapping: thread ID → array index

**Performance Considerations:**
- Memory-bound: limited by bandwidth, not compute
- Coalescing critical: consecutive threads access consecutive memory
- Occupancy matters: need many warps to hide memory latency

---

## 3.3 Pattern 2: Reduce (Aggregation)

**Reduce** combines N elements into a single value using an associative operation (sum, max, min, product, etc.).

### The Challenge
Sequential code processes linearly:
```cpp
int sum = 0;
for (int i = 0; i < N; i++) {
    sum += array[i];  // Each iteration depends on previous
}
```

This has a **dependency chain**—you need result of iteration i before starting iteration i+1. Cannot parallelize naively.

### The Solution: Tree Reduction
Instead of summing linearly (N steps), sum in a tree pattern (log₂N steps):

```
Step 0: [1, 2, 3, 4, 5, 6, 7, 8]
Step 1: [3,    7,    11,   15   ]  // Pair-wise sums
Step 2: [10,        26          ]  // Sum pairs of pairs
Step 3: [36                     ]  // Final sum
```

**Why This Works:**
- Each step processes pairs **independently in parallel**
- Number of active elements halves each step
- Total steps: log₂N instead of N
- Speedup: O(N/log₂N) with N processors

**Implementation Strategy:**
Use shared memory for each block's partial reduction, then reduce partial results on CPU or in a second kernel.

---

## 3.4 Pattern 3: Scan (Prefix Sum)

**Scan** computes cumulative operations: each output element is the operation applied to all previous elements.

### Inclusive Scan (Cumulative Sum)
```
Input:  [1, 2, 3, 4, 5]
Output: [1, 3, 6, 10, 15]
```
`output[i] = input[0] + input[1] + ... + input[i]`

### Exclusive Scan
```
Input:  [1, 2, 3, 4, 5]
Output: [0, 1, 3, 6, 10]
```
`output[i] = input[0] + input[1] + ... + input[i-1]`

### Why Scan Matters
Prefix sums appear everywhere:
- **Stream compaction**: remove elements from arrays
- **Radix sort**: compute output positions
- **Work distribution**: partition work among threads
- **Parallel allocation**: compute offsets for variable-size outputs

### The Challenge
Each element depends on all previous elements—looks sequential. But there's a parallel algorithm.

### Hillis-Steele Scan (Work-Inefficient but Simple)
```
Step 0: [1, 2, 3, 4, 5, 6, 7, 8]
Step 1: [1, 3, 5, 7, 9, 11, 13, 15]  // Add element 1 position left
Step 2: [1, 3, 6, 10, 14, 18, 22, 26] // Add element 2 positions left
Step 3: [1, 3, 6, 10, 15, 21, 28, 36] // Add element 4 positions left
```

Each step doubles the "look-back" distance. Total steps: log₂N.

**Trade-off:** Does O(N log N) work (vs O(N) sequential), but completes in O(log N) time with N processors.

### Blelloch Scan (Work-Efficient)
More complex but does O(N) total work like sequential code. Uses two phases: up-sweep (reduce tree) and down-sweep (distribute sums).

**GPU Implementation:** Use shared memory for block-local scans, then scan block sums, then add block sums back to each block's elements.

---

## 3.5 Pattern 4: Stencil (Neighborhood Operations)

**Stencil** computes each output element from a neighborhood of input elements.

### Examples
- **1D stencil**: 3-point average `output[i] = (input[i-1] + input[i] + input[i+1]) / 3`
- **2D stencil**: Convolution, image blur, finite difference methods
- **3D stencil**: Physics simulations, heat diffusion

### Characteristics
- Each output depends on **multiple inputs** (the stencil)
- Multiple threads **read the same input** (data reuse)
- Boundary conditions require special handling

### The Shared Memory Opportunity
Without shared memory:
```cpp
// Each thread loads 3 values from global memory
output[i] = (input[i-1] + input[i] + input[i+1]) / 3;
```

For a block of 256 threads, this is ~768 global memory loads. But many loads are redundant—thread i loads `input[i-1]`, thread i+1 also loads `input[i-1]` (as its middle element).

With shared memory:
```cpp
// Load block + halo into shared memory (258 loads)
// Each thread reads from shared memory (fast)
// Total: 258 global loads instead of 768
```

**Halo Regions:** Threads at block edges need elements from adjacent blocks. Load these extra "halo" elements into shared memory.

**Why This Works:**
- Global memory load happens once per element (coalesced)
- Multiple threads reuse data from fast shared memory
- 3x reduction in global memory traffic for 3-point stencil
- Larger stencils → more reuse → bigger speedup

---

## 3.6 Algorithmic Transformation Principles

### Principle 1: Expose Parallelism
Find operations that can execute simultaneously. Look for:
- Independent iterations in loops
- Independent function calls
- Data that can be partitioned

### Principle 2: Minimize Dependencies
Restructure algorithms to reduce dependency chains:
- Linear accumulation → tree reduction
- Sequential scan → parallel scan
- Wavefront methods for dependent computations

### Principle 3: Optimize Memory Access
Most GPU algorithms are memory-bound:
- Coalesce global memory accesses
- Reuse data through shared memory
- Minimize global memory traffic

### Principle 4: Balance Work Distribution
Avoid load imbalance:
- Equal work per thread (if possible)
- Dynamic work distribution for irregular problems
- Avoid early exits that waste warp cycles

### Principle 5: Think in Blocks
Design for block-local cooperation:
- What can threads in a block share?
- What requires block-level synchronization?
- How do blocks combine results?

---

## 3.7 Data Dependency Analysis

Before parallelizing, analyze dependencies:

### Read-After-Write (True Dependency)
```cpp
x = compute();
y = use(x);  // Must wait for x
```
Cannot parallelize across this boundary. Must execute in order.

### Write-After-Read (Anti-Dependency)
```cpp
y = use(x);
x = compute();  // Overwrites x after reading
```
Can parallelize if you make private copies or rename variables.

### Write-After-Write (Output Dependency)
```cpp
x = compute1();
x = compute2();  // Both write to x
```
Can parallelize if you separate output locations or use atomics.

### Loop-Carried Dependencies
```cpp
for (int i = 1; i < N; i++) {
    array[i] = array[i] + array[i-1];  // Depends on previous iteration
}
```
This is a scan—requires specialized parallel algorithm.

**Independence Test:** Can I execute iteration i and iteration j in any order without changing the result? If yes → parallelizable.

---

## 3.8 Handling Irregularity

Many real algorithms have irregular patterns:

### Irregular Work Distribution
Some threads do more work than others:
```cpp
for (int i = start[threadIdx.x]; i < end[threadIdx.x]; i++) {
    process(data[i]);
}
```

**Solution:** Partition work more evenly, or use dynamic work distribution (work queues).

### Irregular Memory Access
Random or indirect array access breaks coalescing:
```cpp
output[i] = input[indices[i]];  // indices[i] unpredictable
```

**Solution:** Sort by access pattern, or use texture memory (hardware-cached for random access).

### Conditional Execution (Warp Divergence)
```cpp
if (condition[i]) {
    expensive_operation();
}
```

**Solution:** Partition threads by path (stream compaction), or accept divergence if work is balanced.

---

## 3.9 Multi-Level Parallelism

Complex algorithms often require multiple levels:

### Level 1: Thread-Level
Each thread processes one element:
```cpp
result[i] = f(input[i]);
```

### Level 2: Block-Level
Threads in a block cooperate on a subproblem:
```cpp
__shared__ float block_sum;
// Threads collectively compute block_sum
```

### Level 3: Grid-Level
Multiple kernel launches or atomic operations:
```cpp
kernel_phase1<<<blocks, threads>>>();
kernel_phase2<<<blocks, threads>>>();
```

**Example: Matrix Multiplication**
- Thread level: compute one output element
- Block level: cooperate to load tile into shared memory
- Grid level: partition matrix into tiles across blocks

---

## 3.10 When NOT to Use GPU

GPUs aren't always faster. Avoid GPU for:

### Small Problems
```cpp
N = 100;  // Transfer overhead >> compute time
```
PCIe transfer time dominates. CPU is faster.

### Sequential Algorithms
Algorithms with inherent sequential steps (dynamic programming with dependencies, recursive algorithms without independent subproblems):
```cpp
fib(n) = fib(n-1) + fib(n-2);  // Each call depends on previous
```

### Unpredictable Control Flow
Excessive branching with unpredictable conditions:
```cpp
if (very_complex_condition[i]) {
    path_A();
} else {
    path_B();
}
```
If conditions vary wildly across threads, warp divergence kills performance.

### Low Arithmetic Intensity
Operations that do minimal computation per memory access:
```cpp
output[i] = input[i];  // Just copy
```
Memory bandwidth limits performance—CPU might be comparable.

---

## Key Takeaways

- **Four fundamental patterns**: Map (element-wise), Reduce (aggregation), Scan (prefix sum), Stencil (neighborhood)
- **Decomposition is key**: Break problems into independent subproblems
- **Dependencies limit parallelism**: Analyze and restructure to minimize
- **Memory patterns matter**: Coalescing and reuse determine performance
- **Block-level cooperation**: Use shared memory for data sharing within blocks
- **Multi-level thinking**: Thread, block, and grid-level parallelism
- **Not everything parallelizes**: Understand when CPU is better
