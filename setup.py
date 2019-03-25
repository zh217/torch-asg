from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='torch_asg',
    version='',
    packages=['torch_asg'],
    ext_modules=[CppExtension(name='torch_asg_native',
                              sources=['torch_asg/native/torch_asg.cpp'],
                              extra_compile_args=['-Wno-sign-compare', '-fopenmp'])],
    cmdclass={'build_ext': BuildExtension},
    url='',
    license='',
    author='Ziyang Hu',
    author_email='',
    description=''
)
