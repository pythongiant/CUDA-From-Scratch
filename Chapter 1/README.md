# Chapter 1: Why GPUs Exist and What CUDA Is

Welcome! By the end of this chapter, you'll understand why your gaming GPU can also do serious computing work. No technical background needed - we'll use simple analogies.

## The Big Problem Modern Computing Hit

For years, computers got faster by making CPUs run at higher speeds. But around 2005, this stopped working. CPUs couldn't get any faster without melting or using insane amounts of power.

At the same time, people needed to solve bigger and bigger problems:
- Processing millions of pixels in games/graphics
- Training AI models with billions of calculations
- Simulating weather, physics, or financial markets

These problems all have something in common: **doing the same simple operation on huge amounts of data**.

## CPUs vs GPUs: A Simple Analogy

Imagine you're running a restaurant:

### CPU = Skilled Chef
- One highly trained chef
- Can cook anything: complex recipes, handle special requests
- Very flexible but expensive and slow for simple tasks
- Good for: custom orders, complex dishes

### GPU = Assembly Line Workers
- Hundreds of simple workers
- Each does one basic task (chop vegetables, flip burgers)
- Can't handle complex recipes, but incredibly fast at repetitive work
- Good for: making 1000 identical burgers quickly

**GPUs don't make one task fast. They make many identical tasks fast simultaneously.**

## What CUDA Actually Is

CUDA is NVIDIA's system for writing programs that run on GPUs. It lets you:

1. Write normal C/C++ code
2. Mark certain functions to run on the GPU
3. Tell the GPU to run that function thousands of times in parallel

CUDA gives you **direct control** over the GPU. You decide exactly how work gets divided up and executed.

## The Two-World Model

CUDA programs have two parts:

### CPU Side (Host)
- Your main program runs here
- Controls what happens
- Sends work to the GPU

### GPU Side (Device)
- Does the parallel work
- Has its own memory
- Can't talk directly to the CPU

**You must explicitly copy data between CPU and GPU memory.**

## Your First CUDA Concept: The Kernel

A **kernel** is a function that runs on the GPU. When you launch a kernel, you tell the GPU:

*"Run this function 10,000 times, each time with different data"*

Each run of the function is a **thread**. Threads are grouped into **blocks**, and blocks are arranged in a **grid**.

In CUDA, kernels are marked with the `__global__` keyword. This tells the compiler that the function can be called from the host (CPU) code but will execute on the device (GPU). The double underscores indicate it's a CUDA-specific extension to C/C++, distinguishing kernel functions from regular functions that run on the CPU.

Don't worry about the details yet - we'll see this in action in the next chapter.

## Why CUDA Feels Different

CUDA doesn't automatically make your code parallel. You have to:

- Design your algorithm to work in parallel
- Manage memory transfers between CPU/GPU
- Handle synchronization between parallel tasks

This makes CUDA more work than automatic systems, but you get **predictable, high performance**.

## When CUDA Makes Sense

Use CUDA when you have:
- **Lots of data** (millions/billions of elements)
- **Simple operations** on each element
- **Independent work** (one element doesn't depend on others)

Don't use CUDA for:
- Small amounts of work
- Complex, branching logic
- Problems that are naturally sequential

## Try It Yourself

This chapter doesn't have code yet - we're building concepts first. But here's what we'll do in the next chapter:

```cpp
// A simple kernel that adds 1 to each element
__global__ void addOne(int* data, int N) {
    int i = /* figure out which element this thread should handle */;
    if (i < N) {
        data[i] = data[i] + 1;
    }
}
```

In Chapter 2, we'll run this code and see how it works!

## Key Takeaways

- **GPUs are great at repetitive work on lots of data**
- **CUDA lets you control the GPU directly**
- **Programs run in two worlds: CPU and GPU**
- **You must manage data movement yourself**
- **Performance comes from smart parallel design**

Ready for your first CUDA program? Let's go to Chapter 2!
