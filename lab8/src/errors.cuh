#ifndef ERRORS_CUH
#define ERRORS_CUH

#include <cuda_runtime.h>

#include <iostream>

#define FATAL(description)                                      \
    do {                                                        \
        std::cerr << "Error in " << __FILE__ << ":" << __LINE__ \
                  << ". Message: " << description << std::endl; \
        exit(0);                                                \
    } while (0)

#define CHECK_CALL_ERRORS(call)             \
    do {                                    \
        cudaError_t res = call;             \
        if (res != cudaSuccess) {           \
            FATAL(cudaGetErrorString(res)); \
        }                                   \
    } while (0)

#define CHECK_KERNEL_ERRORS() CHECK_CALL_ERRORS(cudaGetLastError());

#endif