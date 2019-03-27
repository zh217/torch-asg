from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

try:
    import torch
except:
    raise RuntimeError('Please install pytorch>=1 first')

ext_mods = [CppExtension(name='torch_asg_native',
                         sources=['torch_asg/native/torch_asg.cpp'],
                         extra_compile_args=['-fopenmp'])]

if torch.cuda.is_available():
    ext_mods.append(CUDAExtension(name='torch_asg_cuda',
                                  sources=['torch_asg/native/torch_asg_cuda.cpp',
                                           'torch_asg/native/torch_asg_cuda_kernel.cu']))

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
