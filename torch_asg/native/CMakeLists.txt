# cmake -DCMAKE_PREFIX_PATH=/home/zh217/winhome/libtorch_cpu -DCMAKE_TOOLCHAIN_FILE=/home/zh217/vcpkg/scripts/buildsystems/vcpkg.cmake ..

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(torch_asg)

find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(torch_asg torch_asg.cpp)

set(PYBIND11_CPP_STANDARD -std=c++14)

target_link_libraries(torch_asg PRIVATE "${TORCH_LIBRARIES}")