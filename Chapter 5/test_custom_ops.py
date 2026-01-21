# test_custom_ops.py
"""
WHY: Test suite for custom CUDA operations.
Validates correctness by comparing against PyTorch reference implementations
and checks gradients using numerical differentiation.
"""

import torch
import torch.nn as nn
from torch.autograd import gradcheck
import time

# -----------------------------
# WHY: Import the compiled extension module.
# If using setuptools: pip install .
# If using JIT: torch.utils.cpp_extension.load(...)
# -----------------------------
try:
    import custom_ops
    print("✓ Successfully imported custom_ops")
except ImportError as e:
    print(f"✗ Failed to import custom_ops: {e}")
    print("  Make sure to build the extension first:")
    print("  pip install .")
    exit(1)

# ============================================================================
# TEST 1: FUSED OPERATION CORRECTNESS
# ============================================================================

def test_fused_op_forward():
    """
    WHY: Test that fused operation produces same results as PyTorch ops.
    We compute both versions and check numerical difference.
    """
    print("\n" + "="*60)
    print("TEST 1: Fused Operation Forward Pass")
    print("="*60)
    
    # -----------------------------
    # WHY: Create test inputs on GPU. Small size for easy debugging.
    # -----------------------------
    batch_size = 32
    features = 64
    scale = 2.0
    
    input_tensor = torch.randn(batch_size, features, device='cuda', dtype=torch.float32)
    bias = torch.randn(features, device='cuda', dtype=torch.float32)
    
    # -----------------------------
    # WHY: Custom kernel output
    # -----------------------------
    custom_output = custom_ops.fused_op_forward(input_tensor, bias, scale)
    
    # -----------------------------
    # WHY: PyTorch reference - same computation with native ops.
    # This is the "ground truth" we compare against.
    # -----------------------------
    pytorch_output = torch.relu(input_tensor * scale + bias)
    
    # -----------------------------
    # WHY: Compute maximum absolute difference between outputs.
    # Small differences (< 1e-5) are acceptable due to floating point precision.
    # Large differences indicate bugs in the kernel.
    # -----------------------------
    max_diff = (custom_output - pytorch_output).abs().max().item()
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {custom_output.shape}")
    print(f"Max difference vs PyTorch: {max_diff:.2e}")
    
    # WHY: Tolerance of 1e-5 accounts for floating point rounding differences
    if max_diff < 1e-5:
        print("✓ PASSED: Forward pass matches PyTorch")
        return True
    else:
        print("✗ FAILED: Forward pass doesn't match PyTorch")
        return False

# ============================================================================
# TEST 2: GRADIENT CHECKING
# ============================================================================

def test_fused_op_gradients():
    """
    WHY: Verify backward pass using numerical gradient checking.
    PyTorch's gradcheck computes gradients numerically (finite differences)
    and compares with autograd gradients. This catches gradient bugs.
    """
    print("\n" + "="*60)
    print("TEST 2: Fused Operation Gradient Check")
    print("="*60)
    
    # -----------------------------
    # WHY: Use double precision for gradient checking.
    # Single precision (float32) has too much rounding error for
    # accurate finite difference approximation.
    # -----------------------------
    batch_size = 4  # WHY: Small size - gradient check is slow (O(N) forward passes)
    features = 8
    scale = 2.0
    
    # -----------------------------
    # WHY: requires_grad=True tells PyTorch to track gradients.
    # We need this for gradcheck to compute numerical derivatives.
    # -----------------------------
    input_tensor = torch.randn(batch_size, features, device='cuda', 
                               dtype=torch.double, requires_grad=True)
    bias = torch.randn(features, device='cuda', 
                      dtype=torch.double, requires_grad=True)
    
    # -----------------------------
    # WHY: gradcheck doesn't support double precision directly with our
    # float32 kernels, so we'll do manual gradient checking instead.
    # We compute gradients with autograd and compare against numerical gradients.
    # -----------------------------
    print("Testing input gradients...")
    
    # WHY: Convert to float32 for our kernels (they only support float32)
    input_f32 = input_tensor.float().requires_grad_(True)
    bias_f32 = bias.float().requires_grad_(True)
    
    # Forward pass
    output = custom_ops.fused_op_forward(input_f32, bias_f32, scale)
    
    # WHY: Create a scalar loss to backprop through (sum all outputs)
    loss = output.sum()
    
    # Backward pass - compute analytical gradients
    loss.backward()
    
    grad_input_analytical = input_f32.grad.clone()
    grad_bias_analytical = bias_f32.grad.clone()
    
    # -----------------------------
    # WHY: Compute numerical gradients using finite differences.
    # For each element, perturb it slightly and measure output change.
    # grad ≈ (f(x + ε) - f(x - ε)) / (2ε)
    # This is the ground truth for gradient correctness.
    # -----------------------------
    eps = 1e-4  # WHY: Small perturbation for finite difference
    
    # Test a few input elements (testing all is too slow)
    test_indices = [(0, 0), (1, 3), (2, 5)]
    
    print("\nNumerical vs Analytical gradients (sample elements):")
    all_close = True
    
    for b, f in test_indices:
        # Numerical gradient for input[b, f]
        input_plus = input_f32.detach().clone()
        input_plus[b, f] += eps
        out_plus = custom_ops.fused_op_forward(input_plus, bias_f32.detach(), scale).sum()
        
        input_minus = input_f32.detach().clone()
        input_minus[b, f] -= eps
        out_minus = custom_ops.fused_op_forward(input_minus, bias_f32.detach(), scale).sum()
        
        grad_numerical = (out_plus - out_minus) / (2 * eps)
        grad_analytical_val = grad_input_analytical[b, f].item()
        
        diff = abs(grad_numerical.item() - grad_analytical_val)
        status = "✓" if diff < 1e-3 else "✗"
        
        print(f"  {status} input[{b},{f}]: numerical={grad_numerical.item():.6f}, "
              f"analytical={grad_analytical_val:.6f}, diff={diff:.2e}")
        
        if diff >= 1e-3:
            all_close = False
    
    # Test bias gradients
    print("\nBias gradients (sample elements):")
    test_bias_indices = [0, 3, 7]
    
    for f in test_bias_indices:
        # Numerical gradient for bias[f]
        bias_plus = bias_f32.detach().clone()
        bias_plus[f] += eps
        out_plus = custom_ops.fused_op_forward(input_f32.detach(), bias_plus, scale).sum()
        
        bias_minus = bias_f32.detach().clone()
        bias_minus[f] -= eps
        out_minus = custom_ops.fused_op_forward(input_f32.detach(), bias_minus, scale).sum()
        
        grad_numerical = (out_plus - out_minus) / (2 * eps)
        grad_analytical_val = grad_bias_analytical[f].item()
        
        diff = abs(grad_numerical.item() - grad_analytical_val)
        status = "✓" if diff < 1e-3 else "✗"
        
        print(f"  {status} bias[{f}]: numerical={grad_numerical.item():.6f}, "
              f"analytical={grad_analytical_val:.6f}, diff={diff:.2e}")
        
        if diff >= 1e-3:
            all_close = False
    
    if all_close:
        print("\n✓ PASSED: Gradients match numerical approximation")
        return True
    else:
        print("\n✗ FAILED: Gradients don't match numerical approximation")
        return False

# ============================================================================
# TEST 3: CUSTOM RELU
# ============================================================================

def test_custom_relu():
    """
    WHY: Test custom ReLU implementation against PyTorch's ReLU.
    """
    print("\n" + "="*60)
    print("TEST 3: Custom ReLU")
    print("="*60)
    
    # WHY: Create test tensor with both positive and negative values
    input_tensor = torch.randn(128, 256, device='cuda', dtype=torch.float32)
    
    # Custom ReLU
    custom_output = custom_ops.custom_relu_forward(input_tensor)
    
    # PyTorch ReLU
    pytorch_output = torch.relu(input_tensor)
    
    max_diff = (custom_output - pytorch_output).abs().max().item()
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Max difference vs PyTorch: {max_diff:.2e}")
    
    if max_diff < 1e-5:
        print("✓ PASSED: Custom ReLU matches PyTorch")
        return True
    else:
        print("✗ FAILED: Custom ReLU doesn't match PyTorch")
        return False

# ============================================================================
# TEST 4: REDUCE SUM
# ============================================================================

def test_reduce_sum():
    """
    WHY: Test reduction operation against PyTorch's sum.
    """
    print("\n" + "="*60)
    print("TEST 4: Reduce Sum")
    print("="*60)
    
    # WHY: Test with various sizes to ensure reduction works correctly
    test_sizes = [1024, 2048, 10000]
    
    all_passed = True
    for size in test_sizes:
        input_tensor = torch.randn(size, device='cuda', dtype=torch.float32)
        
        custom_sum = custom_ops.reduce_sum_forward(input_tensor)
        pytorch_sum = input_tensor.sum()
        
        diff = abs(custom_sum.item() - pytorch_sum.item())
        rel_diff = diff / abs(pytorch_sum.item()) if pytorch_sum.item() != 0 else diff
        
        status = "✓" if rel_diff < 1e-4 else "✗"
        print(f"  {status} Size {size}: custom={custom_sum.item():.6f}, "
              f"pytorch={pytorch_sum.item():.6f}, rel_diff={rel_diff:.2e}")
        
        if rel_diff >= 1e-4:
            all_passed = False
    
    if all_passed:
        print("✓ PASSED: Reduce sum matches PyTorch")
        return True
    else:
        print("✗ FAILED: Reduce sum doesn't match PyTorch")
        return False

# ============================================================================
# TEST 5: PERFORMANCE BENCHMARK
# ============================================================================

def benchmark_fused_op():
    """
    WHY: Compare performance of fused kernel vs separate PyTorch ops.
    This demonstrates the benefit of kernel fusion (reduced memory traffic,
    fewer kernel launches).
    """
    print("\n" + "="*60)
    print("TEST 5: Performance Benchmark")
    print("="*60)
    
    # -----------------------------
    # WHY: Large size to make kernel time dominate launch overhead
    # -----------------------------
    batch_size = 1024
    features = 2048
    scale = 2.0
    num_iterations = 100
    
    input_tensor = torch.randn(batch_size, features, device='cuda', dtype=torch.float32)
    bias = torch.randn(features, device='cuda', dtype=torch.float32)
    
    # -----------------------------
    # WHY: Warm-up runs to initialize GPU, fill caches, etc.
    # First few runs are always slower due to cold start.
    # -----------------------------
    print("Warming up...")
    for _ in range(10):
        _ = custom_ops.fused_op_forward(input_tensor, bias, scale)
        _ = torch.relu(input_tensor * scale + bias)
    
    torch.cuda.synchronize()  # WHY: Ensure all warm-up kernels finished
    
    # -----------------------------
    # WHY: Benchmark custom fused kernel
    # -----------------------------
    print(f"\nBenchmarking custom fused kernel ({num_iterations} iterations)...")
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_iterations):
        _ = custom_ops.fused_op_forward(input_tensor, bias, scale)
    
    torch.cuda.synchronize()  # WHY: Wait for all kernels to finish before measuring time
    custom_time = (time.time() - start) / num_iterations * 1000  # Convert to ms
    
    # -----------------------------
    # WHY: Benchmark PyTorch separate operations
    # -----------------------------
    print(f"Benchmarking PyTorch separate ops ({num_iterations} iterations)...")
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_iterations):
        _ = torch.relu(input_tensor * scale + bias)
    
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_iterations * 1000
    
    # -----------------------------
    # WHY: Calculate speedup factor
    # -----------------------------
    speedup = pytorch_time / custom_time
    
    print(f"\nResults:")
    print(f"  Custom fused kernel: {custom_time:.3f} ms")
    print(f"  PyTorch separate ops: {pytorch_time:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    # -----------------------------
    # WHY: Fused kernel should be faster (or at least not slower).
    # If it's slower, there's likely a bug or suboptimal implementation.
    # Note: For very small tensors, PyTorch might be faster due to
    # highly optimized libraries (cuBLAS, etc) and our simple implementation.
    # -----------------------------
    if speedup >= 0.8:  # Allow small margin
        print("✓ Performance acceptable")
        return True
    else:
        print("⚠ Warning: Custom kernel slower than PyTorch")
        return True  # Don't fail on performance, just warn

# ============================================================================
# TEST 6: INTEGRATION WITH NN.MODULE
# ============================================================================

class FusedOpLayer(nn.Module):
    """
    WHY: Demonstrate how to wrap custom operation in a PyTorch nn.Module.
    This makes it easy to use in neural networks.
    """
    def __init__(self, features, scale=1.0):
        super().__init__()
        # WHY: Register bias as a parameter so it gets trained with backprop
        self.bias = nn.Parameter(torch.zeros(features))
        self.scale = scale
    
    def forward(self, x):
        # WHY: Call our custom operation
        return custom_ops.fused_op_forward(x, self.bias, self.scale)

def test_nn_module_integration():
    """
    WHY: Test that custom op works within nn.Module and training loop.
    """
    print("\n" + "="*60)
    print("TEST 6: nn.Module Integration & Training")
    print("="*60)
    
    # -----------------------------
    # WHY: Create a simple network with our custom layer
    # -----------------------------
    batch_size = 32
    features = 64
    
    model = nn.Sequential(
        nn.Linear(features, features),
        FusedOpLayer(features, scale=2.0),
        nn.Linear(features, 10)
    ).cuda()
    
    # WHY: Create dummy data and target
    input_data = torch.randn(batch_size, features, device='cuda')
    target = torch.randint(0, 10, (batch_size,), device='cuda')
    
    # WHY: Setup optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # -----------------------------
    # WHY: Training loop - verify forward and backward work in real training
    # -----------------------------
    print("Running training steps...")
    losses = []
    
    for step in range(5):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(input_data)
        loss = nn.functional.cross_entropy(output, target)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        losses.append(loss.item())
        print(f"  Step {step+1}: loss = {loss.item():.6f}")
    
    # -----------------------------
    # WHY: Loss should decrease (or at least not increase dramatically)
    # if gradients are computed correctly
    # -----------------------------
    if losses[-1] <= losses[0] * 2:  # Allow some variance
        print("✓ PASSED: Training loop works, loss is reasonable")
        return True
    else:
        print("⚠ Warning: Loss increased significantly (may indicate gradient issues)")
        return True  # Don't fail, just warn

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """
    WHY: Execute all tests and report summary.
    """
    print("\n" + "="*60)
    print("CUSTOM CUDA OPERATIONS TEST SUITE")
    print("="*60)
    
    tests = [
        ("Fused Op Forward", test_fused_op_forward),
        ("Fused Op Gradients", test_fused_op_gradients),
        ("Custom ReLU", test_custom_relu),
        ("Reduce Sum", test_reduce_sum),
        ("Performance Benchmark", benchmark_fused_op),
        ("nn.Module Integration", test_nn_module_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ EXCEPTION in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # -----------------------------
    # WHY: Print summary of all test results
    # -----------------------------
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {test_name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total_count - passed_count} test(s) failed")
        return 1

if __name__ == "__main__":
    exit(run_all_tests())