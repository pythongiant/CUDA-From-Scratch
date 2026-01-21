# setup.py
"""
WHY: setuptools configuration for building PyTorch CUDA extensions.
This file tells Python how to compile C++/CUDA code into a loadable module.
It handles compiler flags, include paths, and linking against PyTorch/CUDA libraries.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# -----------------------------
# WHY: Define the extension module with all source files.
# CUDAExtension automatically handles CUDA compilation, linking, and
# PyTorch integration. It knows how to find CUDA toolkit, set up nvcc,
# and link against PyTorch's libtorch.
# -----------------------------
setup(
    name='custom_ops',  # WHY: Module name for import (import custom_ops)
    ext_modules=[
        CUDAExtension(
            name='custom_ops',  # WHY: Must match module name
            sources=[
                'custom_ops.cpp',      # WHY: C++ bindings (pybind11 layer)
                'custom_kernels.cu',   # WHY: CUDA kernel implementations
            ],
            extra_compile_args={
                # WHY: C++ compiler flags for optimization and standards
                'cxx': [
                    '-O3',              # WHY: Maximum optimization level
                    '-std=c++14',       # WHY: C++14 required by PyTorch
                ],
                # WHY: NVCC (CUDA compiler) flags for GPU code
                'nvcc': [
                    '-O3',              # WHY: Maximum optimization
                    '--use_fast_math',  # WHY: Trade precision for speed (sqrt, div, etc)
                    '-arch=sm_70',      # WHY: Target GPU architecture (Volta/Turing/Ampere)
                    # Note: Adjust sm_70 based on your GPU (sm_60, sm_75, sm_80, sm_86, etc)
                ],
            },
        )
    ],
    cmdclass={
        # WHY: BuildExtension is PyTorch's custom build command that handles
        # CUDA compilation, mixed C++/CUDA builds, and proper linking.
        'build_ext': BuildExtension
    },
    # WHY: Ensure PyTorch is installed before building (it's required)
    install_requires=['torch'],
)