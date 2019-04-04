from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

try:
    import torch
except:
    raise RuntimeError('Please install pytorch>=1 first')

ext_mods = [CppExtension(name='torch_asg_native',
                         sources=['torch_asg/native/utils.cpp',
                                  'torch_asg/native/force_aligned_lattice.cpp',
                                  'torch_asg/native/fully_connected_lattice.cpp',
                                  'torch_asg/native/extension.cpp'],
                         # extra_compile_args=['-fopenmp', '-Ofast']
                         extra_compile_args=['-fopenmp', '-O0', '-g']
                         )]

# if True or torch.cuda.is_available():
#     ext_mods.append(CUDAExtension(name='torch_asg_cuda',
#                                   sources=['torch_asg/native/torch_asg_cuda.cpp',
#                                            'torch_asg/native/torch_asg_cuda_kernel.cu'],
#                                   extra_compile_args={
#                                       'cxx': ['-O2', ],
#                                       'nvcc': ['--gpu-architecture=sm_70', '-O3', '--use_fast_math',
#                                                '--expt-extended-lambda',
#                                                '--expt-relaxed-constexpr',
#                                                '-I./torch_asg/cub-1.8.0'
#                                                ],
#                                   }))

setup(
    name='torch_asg',
    version='',
    packages=['torch_asg'],
    ext_modules=ext_mods,
    cmdclass={'build_ext': BuildExtension},
    url='',
    license='',
    author='Ziyang Hu',
    author_email='',
    description=''
)
