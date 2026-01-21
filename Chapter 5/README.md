# Chapter 5: Connect CUDA to Python/PyTorch - Build Real Applications

You've learned CUDA fundamentals. Now let's connect this to real Python code! You'll build custom operations that work seamlessly with PyTorch and machine learning.

## What You'll Build

A complete PyTorch extension with:
- Custom CUDA kernels for performance-critical operations
- Automatic gradients for training neural networks
- Python interface that feels like normal PyTorch
- Performance benchmarks showing speedups

## The Big Picture

Your CUDA knowledge → PyTorch extension → Faster ML models

## Quick Start: Run the Examples

First, check your setup:

```bash
# Check CUDA and PyTorch
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
nvcc --version
```

## Method 1: Install the Extension (Easiest)

```bash
cd Chapter\ 5/
pip install .
```

This compiles everything and installs the `custom_ops` module.

## Method 2: JIT Compilation (For Development)

Create `load_extension.py`:

```python
from torch.utils.cpp_extension import load

# Compile and load the extension
custom_ops = load(
    name='custom_ops',
    sources=['custom_ops.cpp', 'custom_kernels.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=True
)

print("Extension loaded successfully!")
```

Run it:
```bash
python load_extension.py
```

## Test It Works

```bash
python test_custom_ops.py
```

You should see all tests pass, including performance benchmarks.

## What the Code Does

### Fused Operation: `relu(input * scale + bias)`
Combines 3 operations into 1 kernel - saves memory bandwidth and kernel launches.

**In Python:**
```python
import torch
import custom_ops

# Create data on GPU
input_tensor = torch.randn(32, 64, device='cuda', requires_grad=True)
bias = torch.randn(64, device='cuda', requires_grad=True)

# Your custom operation
output = custom_ops.fused_op_forward(input_tensor, bias, scale=2.0)

# Automatic gradients work!
loss = output.sum()
loss.backward()
print("Gradients computed:", input_tensor.grad is not None)
```

### Performance Comparison

The tests show your custom kernel vs PyTorch's separate operations:

```
Custom fused kernel: 0.45 ms
PyTorch separate ops: 0.67 ms
Speedup: 1.5x
```

**Why faster?** One kernel launch instead of three, no intermediate memory writes.

## Build Your Own Operation

Let's add a simple element-wise square operation:

### 1. Add CUDA Kernel (`custom_kernels.cu`)

```cpp
__global__ void square_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    output[idx] = input[idx] * input[idx];
}

torch::Tensor square_cuda(torch::Tensor input) {
    const int N = input.numel();
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    square_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                       output.data_ptr<float>(), N);
    return output;
}
```

### 2. Add C++ Binding (`custom_ops.cpp`)

```cpp
torch::Tensor square_cuda(torch::Tensor input);  // Declaration

class SquareFunction : public torch::autograd::Function<SquareFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                torch::Tensor input) {
        TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
        ctx->save_for_backward({input});
        return square_cuda(input);
    }
    
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs)
    {
        auto input = ctx->get_saved_variables()[0];
        auto grad_output = grad_outputs[0];
        // d/dx(x²) = 2x
        auto grad_input = 2.0 * input * grad_output;
        return {grad_input};
    }
};

torch::Tensor square_forward(torch::Tensor input) {
    return SquareFunction::apply(input);
}

// In PYBIND11_MODULE:
m.def("square_forward", &square_forward, "Element-wise square");
```

### 3. Test It

```python
import torch
import custom_ops

x = torch.randn(10, device='cuda', requires_grad=True)
y = custom_ops.square_forward(x)  # y = x²
loss = y.sum()
loss.backward()

print("x:", x)
print("y:", y) 
print("x.grad:", x.grad)  # Should be 2*x
```

## Use in Neural Networks

Wrap your operation in a PyTorch module:

```python
import torch.nn as nn
import custom_ops

class CustomLayer(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(features, features))
        self.bias = nn.Parameter(torch.zeros(features))
    
    def forward(self, x):
        # Use your custom fused operation
        return custom_ops.fused_op_forward(
            torch.matmul(x, self.weight), 
            self.bias, 
            scale=1.0
        )

# Build model
model = nn.Sequential(
    nn.Linear(784, 256),
    CustomLayer(256),  # Your custom CUDA layer
    nn.Linear(256, 10)
).cuda()

# Train normally
optimizer = torch.optim.Adam(model.parameters())
# ... training loop ...
```

## When to Use Custom CUDA

**Use custom CUDA when:**
- PyTorch operations are too slow for your bottleneck
- You need a specific operation not in PyTorch
- Memory bandwidth is the limiting factor
- You want to fuse multiple operations

**Don't use custom CUDA for:**
- Simple operations (PyTorch is already optimized)
- Small tensors (GPU launch overhead dominates)
- Operations that aren't your bottleneck

## Debug Common Issues

### Build Errors
```bash
# CUDA version mismatch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Missing CUDA
export CUDA_HOME=/usr/local/cuda
```

### Runtime Errors
```python
# Tensors not on GPU
x = x.cuda()

# Wrong tensor types
x = x.float()
```

### Performance Issues
```python
# Profile your code
with torch.profiler.profile() as prof:
    # your code
print(prof)
```

## Key Takeaways

- **Extensions connect CUDA to Python** with automatic gradients
- **Fuse operations** for better performance (memory bandwidth savings)
- **Test thoroughly** - gradients, performance, edge cases
- **Start simple** - get one operation working, then add complexity
- **Profile first** - ensure custom CUDA actually helps

## You're Done!

You now have the complete pipeline:
1. **Understand GPU architecture** (Chapters 1-3)
2. **Write parallel algorithms** (Chapter 4)
3. **Connect to real applications** (Chapter 5)

Go build something fast! 🚀
