#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

glm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={
                "nvcc": [
                    f"-I{glm_path}",
                    "-std=c++17",
                    "-lineinfo",
                    "-O3",
                    "--use_fast_math",
                    "-Xptxas=-O3",
                    # Target GPU architecture:
                    # sm_80  -> NVIDIA Ampere data-center GPUs such as A100
                    # sm_86  -> NVIDIA Ampere consumer GPUs such as RTX 3080 / 3090
                    # sm_89  -> NVIDIA Ada Lovelace GPUs such as RTX 4090
                    # sm_90  -> NVIDIA Hopper GPUs such as H100
                    #
                    # Change this flag according to your GPU model.
                    # Example:
                    #   A100   -> -gencode=arch=compute_80,code=sm_80
                    #   RTX4090-> -gencode=arch=compute_89,code=sm_89
                    #   H100   -> -gencode=arch=compute_90,code=sm_90
                    "-gencode=arch=compute_80,code=sm_80"
                ]
            }
            )
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

