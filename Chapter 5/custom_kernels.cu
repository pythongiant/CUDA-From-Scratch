// custom_kernels.cu
/*
 * WHY: CUDA kernel implementations - the actual GPU code.
 * This file contains __global__ kernels (run on GPU) and host functions
 * that launch them. These are called from custom_ops.cpp.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// -----------------------------
// WHY: Macro for CUDA error checking. CUDA functions return error codes,
// but don't throw exceptions. We wrap them to get proper error messages.
// Without this, silent CUDA errors are hard to debug.
// -----------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(                                          \
                std::string("CUDA error at ") + __FILE__ + ":" +               \
                std::to_string(__LINE__) + " - " +                             \
                cudaGetErrorString(err));                                      \
        }                                                                      \
    } while(0)

// ============================================================================
// FUSED OPERATION KERNELS
// ============================================================================

// -----------------------------
// Forward kernel: y = relu(input * scale + bias)
// WHY: Fuse three operations (multiply, add, relu) into one kernel.
// Without fusion, PyTorch would launch 3 kernels and create 2 intermediate
// tensors. This wastes memory bandwidth and adds kernel launch overhead.
// Each thread processes one element - this is the MAP pattern.
// -----------------------------
__global__ void fused_op_forward_kernel(
    const float* __restrict__ input,   // WHY: __restrict__ tells compiler pointers don't alias
    const float* __restrict__ bias,    // WHY: Enables better optimization (register caching, etc)
    float* __restrict__ output,
    int batch_size,
    int features,
    float scale)
{
    // -----------------------------
    // WHY: 2D grid: x dimension for features, y dimension for batch.
    // This maps naturally to 2D tensor [batch, features].
    // Each thread computes one output element.
    // -----------------------------
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    // -----------------------------
    // WHY: Guard against over-launch. 2D grids often don't divide evenly.
    // -----------------------------
    if (feature_idx >= features || batch_idx >= batch_size) return;

    // -----------------------------
    // WHY: Compute 1D index into flattened array.
    // PyTorch stores [batch, features] in row-major order.
    // -----------------------------
    int idx = batch_idx * features + feature_idx;

    // -----------------------------
    // WHY: The actual fused computation. Three operations in one:
    // 1. Multiply by scale
    // 2. Add bias (broadcast across batch dimension)
    // 3. ReLU activation
    // All happen in registers - no intermediate memory writes.
    // fmaxf is GPU intrinsic (faster than max(0.0f, x) on some architectures).
    // -----------------------------
    float val = input[idx] * scale + bias[feature_idx];
    output[idx] = fmaxf(0.0f, val);
}

// -----------------------------
// Backward kernel: compute gradients
// WHY: Chain rule: grad_input = grad_output * scale * (output > 0)
//                  grad_bias = sum over batch of grad_output * (output > 0)
// The (output > 0) part is the ReLU gradient mask.
// -----------------------------
__global__ void fused_op_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ grad_input,
    float* __restrict__ grad_bias_partial,  // WHY: Partial sums per block (for reduction)
    int batch_size,
    int features,
    float scale)
{
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (feature_idx >= features || batch_idx >= batch_size) return;

    int idx = batch_idx * features + feature_idx;

    // -----------------------------
    // WHY: Recompute forward pass to determine ReLU mask.
    // This is RECOMPUTATION - trade compute for memory.
    // Alternative: save mask in forward (uses memory).
    // For simple ops like ReLU, recomputation is usually faster.
    // -----------------------------
    float forward_val = input[idx] * scale + bias[feature_idx];
    float relu_mask = (forward_val > 0.0f) ? 1.0f : 0.0f;

    // -----------------------------
    // WHY: Chain rule for input gradient.
    // ∂L/∂input = ∂L/∂output * ∂output/∂input
    //           = grad_output * scale * relu_mask
    // -----------------------------
    grad_input[idx] = grad_output[idx] * scale * relu_mask;

    // -----------------------------
    // WHY: For bias gradient, we need to sum over batch dimension.
    // Each element of bias affects all batch elements, so we accumulate.
    // This is a REDUCE operation. We do partial reduction here, full
    // reduction happens later (to avoid atomics in this kernel).
    // -----------------------------
    // Store partial bias gradient for this thread
    // (will be reduced later)
    if (batch_idx == 0) {  // WHY: Only first batch thread writes (simplified version)
        float bias_grad = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            int i = b * features + feature_idx;
            float fwd = input[i] * scale + bias[feature_idx];
            float mask = (fwd > 0.0f) ? 1.0f : 0.0f;
            bias_grad += grad_output[i] * mask;
        }
        grad_bias_partial[feature_idx] = bias_grad;
    }
}

// -----------------------------
// Host function: launch forward kernel
// WHY: This is the C++ interface that custom_ops.cpp calls.
// It handles memory allocation, launch configuration, and error checking.
// -----------------------------
torch::Tensor fused_op_cuda_forward(
    torch::Tensor input,
    torch::Tensor bias,
    float scale)
{
    // -----------------------------
    // WHY: Extract dimensions from PyTorch tensor.
    // -----------------------------
    const int batch_size = input.size(0);
    const int features = input.size(1);

    // -----------------------------
    // WHY: Allocate output tensor with same shape and properties as input.
    // torch::empty is like torch.empty in Python - uninitialized memory.
    // We'll write to it in the kernel, so initialization would be wasted work.
    // -----------------------------
    auto output = torch::empty_like(input);

    // -----------------------------
    // WHY: Configure 2D launch grid. We use 16x16 thread blocks (256 threads).
    // This is a good balance for 2D problems - enough parallelism, not too
    // many blocks. Blocks must cover entire [batch, features] space.
    // -----------------------------
    const dim3 threads(16, 16);
    const dim3 blocks(
        (features + threads.x - 1) / threads.x,
        (batch_size + threads.y - 1) / threads.y
    );

    // -----------------------------
    // WHY: Launch kernel. <<<blocks, threads>>> is CUDA launch syntax.
    // data_ptr<float>() gets raw device pointer from PyTorch tensor.
    // -----------------------------
    fused_op_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        features,
        scale
    );

    // -----------------------------
    // WHY: Check for kernel launch errors (configuration errors).
    // -----------------------------
    CUDA_CHECK(cudaGetLastError());

    // -----------------------------
    // WHY: Wait for kernel to complete and check for execution errors.
    // In production, you might skip this for async execution, but it's
    // essential for debugging. Without it, errors surface later, obscuring
    // their source.
    // -----------------------------
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

// -----------------------------
// Host function: launch backward kernel
// WHY: Returns vector of gradients: [grad_input, grad_bias]
// -----------------------------
std::vector<torch::Tensor> fused_op_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor bias,
    float scale)
{
    const int batch_size = input.size(0);
    const int features = input.size(1);

    // WHY: Allocate gradient tensors
    auto grad_input = torch::empty_like(input);
    auto grad_bias = torch::empty_like(bias);

    const dim3 threads(16, 16);
    const dim3 blocks(
        (features + threads.x - 1) / threads.x,
        (batch_size + threads.y - 1) / threads.y
    );

    fused_op_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        grad_bias.data_ptr<float>(),
        batch_size,
        features,
        scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return {grad_input, grad_bias};
}

// ============================================================================
// CUSTOM RELU KERNELS
// ============================================================================

// -----------------------------
// WHY: Simple ReLU for demonstration. max(0, x) per element.
// -----------------------------
__global__ void custom_relu_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    output[idx] = fmaxf(0.0f, input[idx]);
}

// -----------------------------
// WHY: ReLU backward is a mask. Gradient passes through where input > 0.
// -----------------------------
__global__ void custom_relu_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    float* __restrict__ grad_input,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // WHY: Chain rule: grad_input = grad_output * ∂relu/∂input
    //                              = grad_output * (input > 0 ? 1 : 0)
    grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
}

torch::Tensor custom_relu_cuda_forward(torch::Tensor input) {
    const int N = input.numel();
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    custom_relu_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

torch::Tensor custom_relu_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input)
{
    const int N = input.numel();
    auto grad_input = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    custom_relu_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        N
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return grad_input;
}

// ============================================================================
// REDUCE SUM KERNEL
// ============================================================================

// -----------------------------
// WHY: Block-level reduction using shared memory (from Chapter 3).
// Each block produces one partial sum. We then sum partials on CPU
// (for simplicity - production would do hierarchical GPU reduction).
// -----------------------------
__global__ void reduce_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ partial_sums,
    int N)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // WHY: Shared memory for block-local reduction. Much faster than global memory.
    __shared__ float shared_data[256];

    // WHY: Load data into shared memory (or 0 if out of bounds)
    shared_data[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // WHY: Tree reduction in shared memory (see Chapter 3 for detailed explanation)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // WHY: Thread 0 writes block's partial sum
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_data[0];
    }
}

torch::Tensor reduce_sum_cuda(torch::Tensor input) {
    const int N = input.numel();
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    // WHY: Allocate temporary buffer for partial sums (one per block)
    auto partial_sums = torch::empty({blocks}, input.options());

    reduce_sum_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        N
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // WHY: Final reduction on CPU (simple for small number of blocks).
    // Production code would launch another reduction kernel for large N.
    auto h_partials = partial_sums.cpu();
    float total = 0.0f;
    for (int i = 0; i < blocks; i++) {
        total += h_partials[i].item<float>();
    }

    // WHY: Return scalar tensor on GPU to match input device
    return torch::tensor({total}, input.options());
}