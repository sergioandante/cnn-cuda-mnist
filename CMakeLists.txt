cmake_minimum_required(VERSION 3.20)
project(cnn_cuda_mnist LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)

if (MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W3>)
endif()

add_executable(cnn_train
    src/mnist.cpp
    src/kernels.cu
    src/layers.cu
    src/model.cu
    src/train.cu
)

target_include_directories(cnn_train PRIVATE include)
set_target_properties(cnn_train PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(cnn_train PRIVATE CUDA::curand)
