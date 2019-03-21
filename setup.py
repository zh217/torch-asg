from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='torch_asg',
    version='',
    packages=['torch_asg'],
    ext_modules=[CppExtension('torch_asg_native', ['torch_asg/native/torch_asg.cpp'])],
    cmdclass={'build_ext': BuildExtension},
    url='',
    license='',
    author='Ziyang Hu',
    author_email='',
    description=''
)
