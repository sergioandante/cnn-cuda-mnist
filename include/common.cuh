#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "Error CUDA en %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                            \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                        \
    } while (0)

#define CURAND_CHECK(call)                                                       \
    do {                                                                        \
        curandStatus_t st__ = (call);                                           \
        if (st__ != CURAND_STATUS_SUCCESS) {                                    \
            std::fprintf(stderr, "Error cuRAND en %s:%d: codigo %d\n", __FILE__, \
                         __LINE__, (int)st__);                                  \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                        \
    } while (0)

inline float* dalloc(size_t n) {
    float* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
    return p;
}

inline void dzero(float* p, size_t n) {
    CUDA_CHECK(cudaMemset(p, 0, n * sizeof(float)));
}
