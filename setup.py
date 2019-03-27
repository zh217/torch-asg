from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

setup(
    name='torch_asg',
    version='',
    packages=['torch_asg'],
    ext_modules=[CppExtension(name='torch_asg_native',
                              sources=['torch_asg/native/torch_asg.cpp'],
                              extra_compile_args=['-fopenmp']),
                 # CUDAExtension(name='torch_asg_cuda',
                 #               sources=['torch_asg/native/torch_asg_cuda.cpp',
                 #                        'torch_asg/native/torch_asg_cuda_kernel.cu'])
                 ],
    cmdclass={'build_ext': BuildExtension},
    url='',
    license='',
    author='Ziyang Hu',
    author_email='',
    description=''
)
