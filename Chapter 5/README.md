# Chapter 5 — From CUDA Kernels to PyTorch Extensions

This chapter explains **how to integrate custom CUDA kernels into PyTorch**, bridging the gap between low-level GPU programming and high-level deep learning frameworks. You'll learn why custom kernels matter, how PyTorch's extension mechanism works, and how to write operators that integrate seamlessly with autograd.

---

## Project Structure

```
chapter5_pytorch_extensions/
├── README.md                 # This file
├── setup.py                  # Build configuration
├── custom_ops.cpp            # C++ bindings (pybind11 layer)
├── custom_kernels.cu         # CUDA kernel implementations
├── test_custom_ops.py        # Test suite
└── example_usage.py          # Practical examples
```

---

## Prerequisites

### System Requirements
- **CUDA Toolkit**: Version 11.0 or later (must match PyTorch CUDA version)
- **C++ Compiler**: GCC 7+ (Linux), MSVC 2019+ (Windows), Clang (macOS)
- **Python**: 3.8 or later
- **PyTorch**: 1.12 or later with CUDA support

### Check Your Setup

```bash
# Verify CUDA installation
nvcc --version

# Check PyTorch CUDA version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Verify CUDA and PyTorch versions match
# If PyTorch is built with CUDA 11.8, you need CUDA Toolkit 11.8.x installed
```

**Important**: Your CUDA Toolkit version must match PyTorch's CUDA version. If there's a mismatch:
```bash
# Option 1: Install matching PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8

# Option 2: Install matching CUDA Toolkit
# Download from https://developer.nvidia.com/cuda-toolkit-archive
```

---

## Installation

### Method 1: Build and Install (Recommended for Production)

```bash
# Clone or navigate to the chapter directory
cd chapter5_pytorch_extensions/

# Build and install the extension
pip install .

# This compiles the C++/CUDA code and installs the 'custom_ops' module
# Build artifacts are cached, so rebuilds are faster
```

**What happens during build:**
1. `setuptools` reads `setup.py`
2. C++ compiler compiles `custom_ops.cpp`
3. NVCC compiles `custom_kernels.cu`
4. Linker creates shared library (`.so` on Linux, `.pyd` on Windows)
5. Module installed to Python's site-packages

**Common build errors and solutions:**

```bash
# Error: "CUDA_HOME not set"
export CUDA_HOME=/usr/local/cuda  # Linux/Mac
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8  # Windows

# Error: "nvcc not found"
export PATH=$CUDA_HOME/bin:$PATH  # Add CUDA to PATH

# Error: "incompatible CUDA version"
# Install PyTorch matching your CUDA version (see Prerequisites)

# Error: "undefined symbol: _ZN2at..."
# ABI compatibility issue - rebuild PyTorch or use compatible compiler
pip install torch --force-reinstall
```

### Method 2: JIT Compilation (Development/Experimentation)

Create a file `load_jit.py`:

```python
"""
WHY: JIT compilation for rapid development.
Compiles on first import, caches for subsequent runs.
No need to reinstall after code changes - just re-import.
"""

from torch.utils.cpp_extension import load

custom_ops = load(
    name='custom_ops',
    sources=['custom_ops.cpp', 'custom_kernels.cu'],
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_70'],  # Adjust sm_XX for your GPU
    verbose=True  # Show compilation output
)

# Now use custom_ops as normal
import torch
input_tensor = torch.randn(32, 64, device='cuda')
bias = torch.randn(64, device='cuda')
output = custom_ops.fused_op_forward(input_tensor, bias, 2.0)
print("JIT compilation successful!")
```

Run it:
```bash
python load_jit.py
```

**GPU Architecture (sm_XX) for different GPUs:**
- Tesla V100: `sm_70`
- RTX 2080/3080: `sm_75` or `sm_86`
- A100: `sm_80`
- RTX 4090: `sm_89`
- H100: `sm_90`

Find yours:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
# Output: "7.5" means sm_75
```

---

## Running Tests

### Quick Test (Verify Installation)

```bash
# After installation (Method 1) or with JIT (Method 2)
python test_custom_ops.py
```

**Expected output:**
```
✓ Successfully imported custom_ops
==========================================
TEST 1: Fused Operation Forward Pass
==========================================
Input shape: torch.Size([32, 64])
Output shape: torch.Size([32, 64])
Max difference vs PyTorch: 9.54e-07
✓ PASSED: Forward pass matches PyTorch

[... more tests ...]

==========================================
TEST SUMMARY
==========================================
  ✓ PASSED: Fused Op Forward
  ✓ PASSED: Fused Op Gradients
  ✓ PASSED: Custom ReLU
  ✓ PASSED: Reduce Sum
  ✓ PASSED: Performance Benchmark
  ✓ PASSED: nn.Module Integration

Total: 6/6 tests passed

🎉 All tests passed!
```

### Individual Tests

```bash
# Test only forward pass
python -c "from test_custom_ops import test_fused_op_forward; test_fused_op_forward()"

# Test only gradients
python -c "from test_custom_ops import test_fused_op_gradients; test_fused_op_gradients()"

# Run performance benchmark
python -c "from test_custom_ops import benchmark_fused_op; benchmark_fused_op()"
```

### Debugging Failed Tests

If tests fail:

```bash
# 1. Enable verbose CUDA error checking
export CUDA_LAUNCH_BLOCKING=1

# 2. Run with Python warnings
python -W all test_custom_ops.py

# 3. Check GPU memory
nvidia-smi

# 4. Verify CUDA runtime
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 5. Test basic CUDA operation
python -c "import torch; x = torch.randn(10, device='cuda'); print(x.sum())"
```

---

## Example Usage

### Basic Usage

```python
import torch
import custom_ops

# Create input tensors on GPU
batch_size = 128
features = 256

input_tensor = torch.randn(batch_size, features, device='cuda', requires_grad=True)
bias = torch.randn(features, device='cuda', requires_grad=True)
scale = 2.0

# Forward pass: computes relu(input * scale + bias)
output = custom_ops.fused_op_forward(input_tensor, bias, scale)

# Backward pass (automatic via autograd)
loss = output.sum()
loss.backward()

# Gradients available
print(f"Input gradient shape: {input_tensor.grad.shape}")
print(f"Bias gradient shape: {bias.grad.shape}")
```

### Using in Neural Networks

```python
import torch
import torch.nn as nn
import custom_ops

class FusedOpLayer(nn.Module):
    """Custom layer using fused CUDA operation"""
    def __init__(self, features, scale=1.0):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(features))
        self.scale = scale
    
    def forward(self, x):
        return custom_ops.fused_op_forward(x, self.bias, self.scale)

# Build a network
model = nn.Sequential(
    nn.Linear(512, 256),
    FusedOpLayer(256, scale=2.0),  # Custom layer
    nn.Linear(256, 10)
).cuda()

# Train normally
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    # Your training loop
    optimizer.zero_grad()
    output = model(torch.randn(32, 512, device='cuda'))
    loss = output.sum()
    loss.backward()
    optimizer.step()
```

### Performance Comparison

```python
import torch
import custom_ops
import time

batch_size = 1024
features = 2048
scale = 2.0

input_tensor = torch.randn(batch_size, features, device='cuda')
bias = torch.randn(features, device='cuda')

# Warm-up
for _ in range(10):
    _ = custom_ops.fused_op_forward(input_tensor, bias, scale)
    _ = torch.relu(input_tensor * scale + bias)

torch.cuda.synchronize()

# Benchmark custom kernel
num_iters = 100
start = time.time()
for _ in range(num_iters):
    _ = custom_ops.fused_op_forward(input_tensor, bias, scale)
torch.cuda.synchronize()
custom_time = (time.time() - start) / num_iters * 1000

# Benchmark PyTorch
start = time.time()
for _ in range(num_iters):
    _ = torch.relu(input_tensor * scale + bias)
torch.cuda.synchronize()
pytorch_time = (time.time() - start) / num_iters * 1000

print(f"Custom kernel: {custom_time:.3f} ms")
print(f"PyTorch ops:   {pytorch_time:.3f} ms")
print(f"Speedup:       {pytorch_time/custom_time:.2f}x")
```

---

## Understanding the Code

### File Breakdown

**setup.py**: Build configuration
```python
# Tells setuptools how to compile C++/CUDA code
# Specifies compiler flags, include paths, libraries
# CUDAExtension handles CUDA-specific compilation
```

**custom_ops.cpp**: C++ bindings layer
```cpp
// Bridges Python and CUDA
// Contains:
// - torch::autograd::Function classes (autograd integration)
// - Input validation and error checking
// - Pybind11 module definitions
// - Dispatch to CUDA kernels
```

**custom_kernels.cu**: CUDA kernel implementations
```cuda
// The actual GPU code
// Contains:
// - __global__ kernel functions
// - Host launcher functions
// - CUDA memory management
// - Performance-critical computation
```

**test_custom_ops.py**: Comprehensive test suite
```python
# Tests:
# - Numerical correctness (vs PyTorch)
# - Gradient correctness (numerical differentiation)
# - Integration with PyTorch ecosystem
# - Performance benchmarks
```

### Key Concepts Demonstrated

1. **Kernel Fusion**: Combining multiple operations (multiply, add, relu) into one kernel
   - Reduces memory bandwidth (no intermediate tensors)
   - Reduces kernel launch overhead

2. **Autograd Integration**: Custom forward and backward passes
   - Saves tensors needed for gradient computation
   - Implements chain rule for backpropagation

3. **Memory Coalescing**: Efficient global memory access patterns
   - Consecutive threads access consecutive memory
   - Maximizes memory bandwidth utilization

4. **Shared Memory**: Fast on-chip memory for data reuse
   - Used in reduction kernel
   - Avoids redundant global memory accesses

---

## Troubleshooting

### Build Issues

**Problem**: `fatal error: torch/extension.h: No such file or directory`
```bash
# Solution: Ensure PyTorch is installed
pip install torch torchvision torchaudio
```

**Problem**: `nvcc fatal: Unsupported gpu architecture 'compute_XX'`
```bash
# Solution: Adjust sm_XX in setup.py to match your GPU
# Find your GPU's compute capability:
nvidia-smi --query-gpu=compute_cap --format=csv
```

**Problem**: `undefined symbol: _ZN3c1012CUDACachingAllocator9allocatorE`
```bash
# Solution: ABI mismatch - rebuild with same compiler as PyTorch
pip install torch --force-reinstall
```

### Runtime Issues

**Problem**: `CUDA error: invalid configuration argument`
```python
# Solution: Launch configuration exceeds GPU limits
# Reduce block size or check thread/block limits:
python -c "import torch; print(torch.cuda.get_device_properties(0))"
```

**Problem**: `RuntimeError: Expected all tensors to be on the same device`
```python
# Solution: Ensure all tensors are on GPU
input_tensor = input_tensor.cuda()
bias = bias.cuda()
```

**Problem**: Slow performance compared to PyTorch
```
# Possible causes:
# 1. Tensor size too small (launch overhead dominates)
# 2. Uncoalesced memory access (check access patterns)
# 3. Not enough occupancy (increase block size)
# 4. Missing --use_fast_math flag in compilation

# Profile with:
python -c "import torch.cuda.profiler as profiler; profiler.start(); # your code; profiler.stop()"
```

### Testing Issues

**Problem**: Gradient check fails
```
# Common causes:
# 1. Numerical precision (use double for gradcheck)
# 2. Incorrect backward implementation
# 3. Not saving required tensors in forward

# Debug:
# - Print intermediate values
# - Compare against numerical gradients manually
# - Test smaller tensor sizes
```

---

## Advanced Usage

### Custom Block/Thread Configuration

Modify launch configuration in `custom_kernels.cu`:

```cpp
// Default: 16x16 threads per block (256 threads)
const dim3 threads(16, 16);

// For larger tensors, try:
const dim3 threads(32, 8);  // 256 threads, different shape

// For memory-bound kernels:
const int threads = 512;  // More threads to hide latency

// For compute-bound kernels:
const int threads = 128;  // Fewer threads, more registers
```

### Adding More Operations

To add a new operation:

1. **Add CUDA kernel** in `custom_kernels.cu`:
```cuda
__global__ void my_new_kernel(const float* input, float* output, int N) {
    // Your implementation
}

torch::Tensor my_new_op_cuda(torch::Tensor input) {
    // Launch kernel
}
```

2. **Add C++ binding** in `custom_ops.cpp`:
```cpp
torch::Tensor my_new_op_cuda(torch::Tensor input);  // Forward declaration

torch::Tensor my_new_op(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    return my_new_op_cuda(input);
}

// In PYBIND11_MODULE:
m.def("my_new_op", &my_new_op, "My new operation");
```

3. **Add test** in `test_custom_ops.py`:
```python
def test_my_new_op():
    input_tensor = torch.randn(128, device='cuda')
    output = custom_ops.my_new_op(input_tensor)
    # Verify correctness
    assert output.shape == input_tensor.shape
```

4. **Rebuild**:
```bash
pip install . --force-reinstall
```

---

## Performance Optimization Tips

1. **Profile first**: Use `nvprof` or `nsys` to identify bottlenecks
```bash
nvprof python test_custom_ops.py
```

2. **Maximize occupancy**: Use CUDA Occupancy Calculator
```bash
# Check occupancy:
python -c "import torch; props = torch.cuda.get_device_properties(0); print(f'Max threads per SM: {props.max_threads_per_multiprocessor}')"
```

3. **Coalesce memory access**: Ensure consecutive threads access consecutive memory

4. **Use shared memory**: For data reuse within a block

5. **Minimize synchronization**: Only use `__syncthreads()` when necessary

6. **Optimize launch configuration**: Experiment with different block sizes

---

## Further Reading

- [PyTorch C++ Extension Tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch Autograd Internals](https://pytorch.org/docs/stable/notes/autograd.html)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) - Profiling tool

---

## Key Takeaways

- **Extensions bridge Python and CUDA**: Three layers (Python → C++/pybind11 → CUDA)
- **torch::Tensor is the interface**: C++ API mirrors Python PyTorch
- **Autograd integration requires forward + backward**: Save what you need, compute gradients correctly
- **Two build methods**: setuptools (production) or JIT (development)
- **Launch configuration from tensors**: Convert shapes to blocks/threads
- **Testing is critical**: Gradient checks, numerical validation, benchmarking
- **Fusion is key optimization**: Combine operations to reduce memory traffic
- **Consider when NOT to optimize**: PyTorch ops are already highly optimized

Custom CUDA kernels give you full control over GPU computation while maintaining PyTorch's ease of use and automatic differentiation. Use them when performance matters and PyTorch ops aren't sufficient—but always profile first to ensure the complexity is justified.
