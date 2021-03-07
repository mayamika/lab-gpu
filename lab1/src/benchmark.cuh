#ifndef BENCHMARK_CUH
#define BENCHMARK_CUH

#include <stdio.h>

#include "error.h"

#ifdef BENCHMARK

#define MEASURE(KERNEL)                                          \
    do {                                                         \
        cudaEvent_t start, end;                                  \
        CHECK_CALL_ERRORS(cudaEventCreate(&start));              \
        CHECK_CALL_ERRORS(cudaEventCreate(&end));                \
        CHECK_CALL_ERRORS(cudaEventRecord(start));               \
        KERNEL;                                                  \
        CHECK_CALL_ERRORS(cudaGetLastError());                   \
        CHECK_CALL_ERRORS(cudaEventRecord(end));                 \
        CHECK_CALL_ERRORS(cudaEventSynchronize(end));            \
        float t;                                                 \
        CHECK_CALL_ERRORS(cudaEventElapsedTime(&t, start, end)); \
        CHECK_CALL_ERRORS(cudaEventDestroy(start));              \
        CHECK_CALL_ERRORS(cudaEventDestroy(end));                \
        fprintf(stderr, "%f\n", t);                              \
    } while (0);

#else

#define MEASURE(KERNEL) KERNEL

#endif

#endif