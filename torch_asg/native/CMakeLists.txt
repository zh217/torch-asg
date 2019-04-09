# cmake -DCMAKE_PREFIX_PATH=/home/zh217/winhome/libtorch_cpu -DCMAKE_TOOLCHAIN_FILE=/home/zh217/vcpkg/scripts/buildsystems/vcpkg.cmake ..

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(torch_asg LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)
find_package(CUDA REQUIRED)


pybind11_add_module(torch_asg
        extension.cpp
        utils.cpp
        force_aligned_lattice.cpp
        fully_connected_lattice.cpp
        force_aligned_lattice_kernel.cu)

#add_library(torch_asg_cuda torch_asg_cuda.cpp torch_asg_cuda_kernel.cu fully_connected_lattice.cpp force_aligned_lattice.cpp force_aligned_lattice.h fully_connected_lattice.h force_aligned_lattice_gpu.cpp force_aligned_lattice_gpu.h fully_connected_lattice_gpu.cpp fully_connected_lattice_gpu.h)
#include_directories(torch_asg_cuda INTERFACE ../cub-1.8.0)

target_compile_options(torch_asg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_60,code=sm_60>)

set(PYBIND11_CPP_STANDARD -std=c++14)

target_compile_definitions(torch_asg PRIVATE TORCH_ASG_SUPPORTS_CUDA)

target_link_libraries(torch_asg PRIVATE "${TORCH_LIBRARIES}")
#target_link_libraries(torch_asg_cuda PRIVATE "${TORCH_LIBRARIES}")