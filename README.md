# CUDA From First Principles  
**A Systems-Level Guide to GPU Programming**

---

## What This Book Is

This book is a **from-scratch, first-principles guide to CUDA programming**.  
It is written for readers who want to **understand GPUs as systems**, not just use them as accelerators.

Rather than starting with APIs or performance tricks, the book builds CUDA **from the ground up**:
- why GPUs exist,
- how CUDA maps code to hardware,
- and how architectural constraints shape correct and performant programs.

This is not a cookbook.  
This is a **mental-model-first** book.

---

## Who This Book Is For

This book is written for:

- Systems programmers who want to understand GPUs properly
- ML researchers writing custom CUDA kernels
- Performance engineers and compiler enthusiasts
- Quantitative researchers and scientific programmers
- Advanced students who are unsatisfied with surface-level CUDA tutorials

You will benefit most if you:
- Are comfortable with C/C++
- Understand basic parallelism
- Want to reason about performance, not guess

This book is **not** aimed at:
- Absolute beginners to programming
- Readers looking for copy-paste solutions
- High-level framework users who do not care about hardware behavior

---

## What Makes This Book Different

Most CUDA resources fall into one of two traps:

1. They start with code before explaining *why* the model exists  
2. They hide hardware realities behind abstraction layers

This book does neither.

Instead, it:
- Explains **why the execution model looks the way it does**
- Treats warps, blocks, and memory hierarchy as *first-class concepts*
- Uses code only to **confirm mental models**, not replace them
- Emphasizes **constraints** over tricks

The goal is not to memorize CUDA —  
the goal is to **think the way CUDA hardware thinks**.

---

## Structure of the Book

The book is organized as a progressive tightening of abstraction.

### Chapter 1 — Why CUDA Exists
- Why CPUs failed to scale
- Throughput vs latency machines
- What problems GPUs are fundamentally good at
- What CUDA is — and what it is not

### Chapter 2 — The CUDA Execution Model
- Threads, warps, blocks, and grids
- Logical parallelism vs physical execution
- Warp divergence and its consequences
- Scheduling, latency hiding, and occupancy
- Why global synchronization does not exist

### Chapter 3 — Memory Is the Program
- GPU memory hierarchy
- Registers, shared memory, global memory
- Memory coalescing and access patterns
- Why most kernels are memory-bound
- How hardware constraints shape algorithms

### Chapter 4 — Algorithm Design Under CUDA Constraints
- Reformulating classical algorithms
- Block-level reductions and multi-kernel workflows
- Tradeoffs between parallelism and synchronization
- Determinism, ordering, and correctness
- When CUDA is the wrong tool

### Chapter 5 — From CUDA to Real Systems
- How CUDA kernels integrate into larger software
- Interfacing with ML frameworks
- Performance measurement and profiling
- Portability and architectural evolution
- Designing kernels that age well

Each chapter is written to stand on its own while reinforcing previous mental models.

---

## How to Read This Book

This book is best read **slowly**.

You are encouraged to:
- Pause and reason about diagrams
- Predict behavior before reading explanations
- Treat code as experimental validation
- Reread sections when intuition feels shaky

If something feels “restrictive” or “unnecessarily low-level,”  
that feeling is usually the lesson.

---

## What You Will Walk Away With

By the end of this book, you should be able to:

- Explain CUDA’s execution model without reference material
- Predict when a kernel will underperform before running it
- Reason about divergence, memory access, and synchronization
- Design GPU algorithms that respect hardware constraints
- Read CUDA code written by others and understand *why* it works

Most importantly, you will stop thinking of CUDA as “GPU C++”  
and start thinking of it as **programming a massively parallel system**.

---

## Status

Chapters 1 through 5 are complete.

The book is intended to grow carefully, not rapidly.  
New material will be added only when it deepens understanding rather than expanding surface area.

---

## Philosophy

> Performance is not a trick.  
> Performance is the absence of misunderstanding.

This book exists to remove misunderstanding.

---

## License and Use

This book is written for learning, reference, and careful study.  
If you build systems, research, or software using ideas from it, that is the intended outcome.

---

## Final Note

CUDA rewards clarity of thought.

This book exists to help you earn that clarity.
# CUDA-From-Scratch
