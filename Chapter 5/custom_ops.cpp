// custom_ops.cpp
/*
 * WHY: C++ layer that bridges Python and CUDA kernels.
 * This file contains pybind11 bindings that make C++ functions callable from Python.
 * It also handles tensor validation, dispatch to CUDA kernels, and error checking.
 */

#include <torch/extension.h>
#include <vector>

// -----------------------------
// Forward declarations for CUDA functions
// WHY: These are implemented in custom_kernels.cu. We declare them here
// so C++ code can call them. The actual implementations are compiled
// separately by nvcc (the CUDA compiler).
// -----------------------------
torch::Tensor fused_op_cuda_forward(
    torch::Tensor input,
    torch::Tensor bias,
    float scale);

std::vector<torch::Tensor> fused_op_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor bias,
    float scale);

torch::Tensor custom_relu_cuda_forward(torch::Tensor input);

torch::Tensor custom_relu_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input);

torch::Tensor reduce_sum_cuda(torch::Tensor input);

// ============================================================================
// FUSED OPERATION: y = relu(input * scale + bias)
// ============================================================================

// -----------------------------
// WHY: Autograd-compatible function using torch::autograd::Function.
// This integrates with PyTorch's automatic differentiation system.
// We must implement forward (computation) and backward (gradient computation).
// -----------------------------
class FusedOpFunction : public torch::autograd::Function<FusedOpFunction> {
public:
    // -----------------------------
    // Forward pass
    // WHY: ctx saves information needed for backward pass.
    // We save input and bias because we need them to compute gradients.
    // The scale is saved as a non-tensor value.
    // -----------------------------
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor input,
        torch::Tensor bias,
        double scale)
    {
        // -----------------------------
        // Input validation
        // WHY: TORCH_CHECK throws Python exceptions if conditions fail.
        // We verify tensors are on GPU, correct dtype, and compatible shapes.
        // Better to fail fast with clear error than get cryptic CUDA errors.
        // -----------------------------
        TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
        TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(input.dim() == 2, "input must be 2D (batch, features)");
        TORCH_CHECK(bias.dim() == 1, "bias must be 1D (features)");
        TORCH_CHECK(input.size(1) == bias.size(0), 
                    "input features must match bias size");

        // -----------------------------
        // WHY: Save tensors for backward pass. PyTorch will keep these alive
        // until backward is called. We need input and bias to compute gradients.
        // -----------------------------
        ctx->save_for_backward({input, bias});
        
        // -----------------------------
        // WHY: Save scalar value (scale) separately from tensors.
        // saved_data is a dictionary for non-tensor values.
        // -----------------------------
        ctx->saved_data["scale"] = scale;

        // -----------------------------
        // WHY: Dispatch to CUDA kernel. The actual GPU computation happens here.
        // This calls into custom_kernels.cu where the __global__ kernel lives.
        // -----------------------------
        auto output = fused_op_cuda_forward(input, bias, static_cast<float>(scale));

        return output;
    }

    // -----------------------------
    // Backward pass
    // WHY: Compute gradients with respect to inputs.
    // PyTorch provides grad_output (gradient of loss w.r.t our output).
    // We must compute grad_input and grad_bias using chain rule.
    // The third return value (None) is for scale, which doesn't need a gradient.
    // -----------------------------
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs)
    {
        // -----------------------------
        // WHY: Retrieve saved tensors from forward pass.
        // These are the inputs we saved with save_for_backward.
        // -----------------------------
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto bias = saved[1];
        auto scale = ctx->saved_data["scale"].toDouble();

        // -----------------------------
        // WHY: grad_outputs[0] is the gradient flowing back from the next layer.
        // This is ∂L/∂output where L is the final loss.
        // We need to compute ∂L/∂input and ∂L/∂bias using chain rule.
        // -----------------------------
        auto grad_output = grad_outputs[0];

        // -----------------------------
        // WHY: Call CUDA backward kernel to compute gradients.
        // Returns a vector: [grad_input, grad_bias]
        // -----------------------------
        auto grads = fused_op_cuda_backward(grad_output, input, bias, scale);

        // -----------------------------
        // WHY: Return gradients in same order as forward inputs.
        // forward takes (input, bias, scale), so we return (grad_input, grad_bias, None).
        // torch::Tensor() is an undefined tensor (represents None in Python).
        // -----------------------------
        return {grads[0], grads[1], torch::Tensor()};
    }
};

// -----------------------------
// WHY: C++ wrapper that Python will call.
// This is what gets exposed through pybind11.
// -----------------------------
torch::Tensor fused_op_forward(
    torch::Tensor input,
    torch::Tensor bias,
    double scale)
{
    return FusedOpFunction::apply(input, bias, scale);
}

// ============================================================================
// CUSTOM RELU (for demonstration)
// ============================================================================

class CustomReluFunction : public torch::autograd::Function<CustomReluFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor input)
    {
        TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
        TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");

        // WHY: Save input for backward. ReLU backward needs input to know which
        // elements were positive (gradient passes through) vs negative (gradient = 0).
        ctx->save_for_backward({input});

        return custom_relu_cuda_forward(input);
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto grad_output = grad_outputs[0];

        // WHY: ReLU gradient is simple: pass through gradient where input > 0, else 0.
        auto grad_input = custom_relu_cuda_backward(grad_output, input);

        return {grad_input};
    }
};

torch::Tensor custom_relu_forward(torch::Tensor input) {
    return CustomReluFunction::apply(input);
}

// ============================================================================
// REDUCE SUM (single output, no spatial dimensions)
// ============================================================================

// -----------------------------
// WHY: Reduction produces a scalar from tensor. Backward broadcasts
// gradient to all inputs (since all inputs contributed to the sum).
// -----------------------------
class ReduceSumFunction : public torch::autograd::Function<ReduceSumFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor input)
    {
        TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
        TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");

        // WHY: Save input shape for backward. We need to know the shape
        // to broadcast the scalar gradient back to all input elements.
        ctx->saved_data["input_shape"] = input.sizes();

        return reduce_sum_cuda(input);
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs)
    {
        // WHY: grad_output is a scalar (gradient of loss w.r.t sum).
        // For sum, gradient is the same for all inputs: ∂sum/∂input[i] = 1.
        // So grad_input[i] = grad_output * 1 = grad_output (broadcast to all).
        auto grad_output = grad_outputs[0];
        auto input_shape = ctx->saved_data["input_shape"].toIntVector();

        // WHY: Expand scalar gradient to input shape. PyTorch handles broadcasting.
        auto grad_input = grad_output.expand(input_shape);

        return {grad_input};
    }
};

torch::Tensor reduce_sum_forward(torch::Tensor input) {
    return ReduceSumFunction::apply(input);
}

// ============================================================================
// PYBIND11 MODULE DEFINITION
// ============================================================================

// -----------------------------
// WHY: This macro creates the Python module interface.
// TORCH_EXTENSION_NAME is defined by PyTorch's build system (from setup.py name).
// Each m.def() exposes a C++ function to Python.
// -----------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &fused_op_forward, 
          "Fused operation: relu(input * scale + bias)");
    
    m.def("custom_relu_forward", &custom_relu_forward,
          "Custom ReLU activation");
    
    m.def("reduce_sum_forward", &reduce_sum_forward,
          "Sum all elements of tensor");
}