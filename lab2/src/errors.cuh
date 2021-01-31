#ifndef ERRORS_CUH
#define ERRORS_CUH

#include <cstdio>

#ifdef __INTELLISENSE__
#define __global__
#define __device__
#define __host__
struct uchar4 {
    int x, y, z, w;
};
#define cudaChannelFormatDesc void*
#define cudaArray void*
#define cudaError_t void*
#define cudaSuccess 0
#define cudaMemcpyHostToDevice 0
#define cudaReadModeElementType int
#define cudaFilterModePoint 0
#define cudaAddressModeClamp 0
cudaError_t cudaGetLastError();
template <typename T>
void* cudaCreateChannelDesc();
void* cudaMallocArray(void*, void*, int, int);
void* cudaFreeArray(void*);
char* cudaGetErrorString(void*);
void* cudaMemcpyToArray(void*, int, int, const void*, int, int);
template <typename T, size_t, typename TT>
struct texture {
    int* addressMode;
    void* channelDesc;
    int filterMode;
    bool normalized;
};
template <typename T>
void* cudaBindTextureToArray(T, void*, void*);
template <typename T>
void* cudaUnbindTexture(T);
struct dim3 {
    dim3(int a, int b){};
};
#endif

#define CHECK_CALL_ERRORS(call)                                        \
    do {                                                               \
        cudaError_t res = call;                                        \
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