# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]
sources = glob.glob('src/*.cpp')+glob.glob('src/*.cu')


setup(
    name='pointnet2',
    version='1.0',
    ext_modules=[
        CUDAExtension(
            name='pointnet2.ext',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2"],
            },
        )
    ],
    # Support ninja
    cmdclass={ 
        'build_ext': BuildExtension
    }

)
