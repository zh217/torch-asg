from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

try:
    import torch
except:
    raise RuntimeError('Please install pytorch>=1 first')

ext_mods = []

if torch.cuda.is_available():
    ext_mods.append(CUDAExtension(name='torch_asg_native',
                                  sources=['torch_asg/native/utils.cpp',
                                           'torch_asg/native/force_aligned_lattice.cpp',
                                           'torch_asg/native/fully_connected_lattice.cpp',
                                           'torch_asg/native/extension.cpp',
                                           'torch_asg/native/force_aligned_lattice_kernel.cu'],
                                  extra_compile_args={
                                      'cxx': ['-O2',
                                              '-DTORCH_ASG_SUPPORTS_CUDA'],
                                      'nvcc': ['-arch=sm_60',
                                               '-gencode=arch=compute_60,code=sm_60',
                                               '-gencode=arch=compute_61,code=sm_61',
                                               '-gencode=arch=compute_70,code=sm_70',
                                               '-gencode=arch=compute_75,code=sm_75',
                                               '-gencode=arch=compute_75,code=compute_75',
                                               '-O3',
                                               '--use_fast_math',
                                               '--expt-extended-lambda',
                                               '--expt-relaxed-constexpr'
                                               ],
                                  }))
else:
    ext_mods.append(CppExtension(name='torch_asg_native',
                                 sources=['torch_asg/native/utils.cpp',
                                          'torch_asg/native/force_aligned_lattice.cpp',
                                          'torch_asg/native/fully_connected_lattice.cpp',
                                          'torch_asg/native/extension.cpp'],
                                 extra_compile_args=['-fopenmp', '-Ofast']
                                 # extra_compile_args=['-fopenmp', '-O0', '-g']
                                 ))

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
