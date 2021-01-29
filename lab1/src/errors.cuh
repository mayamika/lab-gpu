#ifndef ERRORS_CUH
#define ERRORS_CUH

#include <cstdio>

#ifdef __INTELLISENSE__
#define __global__
#define __device__
#define __host__
#endif

#define CHECK_CALL_ERRORS(call)                                        \
    do {                                                               \
        cudaDeviceSynchronize();                                       \
        cudaError_t res = call;                                        \
        cudaDeviceSynchronize();                                       \
        if (res != cudaSuccess) {                                      \
            fprintf(stderr, "ERROR in %s:%d. Message: %s\n", __FILE__, \
                    __LINE__, cudaGetErrorString(res));                \
            exit(0);                                                   \
        }                                                              \
    } while (0);

#define CHECK_KERNEL_ERRORS() CHECK_CALL_ERRORS(cudaGetLastError());

#define FATAL(description)                                                   \
    do {                                                                     \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n", __FILE__, __LINE__, \
                description);                                                \
        exit(0);                                                             \
    } while (0);

#endif