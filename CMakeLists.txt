cmake_minimum_required(VERSION 3.10)
project(dlpack_cuda_class LANGUAGES CXX CUDA)

find_package(pybind11 REQUIRED)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

include_directories("/usr/local/include/dlpack")

pybind11_add_module(dlpack_cuda_class MODULE src/dlpack.cu)

target_link_libraries(dlpack_cuda_class PRIVATE ${CUDA_LIBRARIES})

set_target_properties(dlpack_cuda_class PROPERTIES SUFFIX ".so")
