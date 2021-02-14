#ifndef CLASSIFICATION_CUH
#define CLASSIFICATION_CUH

#include <tuple>
#include <vector>

#include "errors.cuh"
#include "image.cuh"
#include "vector.cuh"

namespace classification {
// constant mem
const int MaxClasses = 32;
__constant__ float4 averages[MaxClasses];

// texture
using _CudaTexture = texture<uchar4, 2, cudaReadModeElementType>;
_CudaTexture source_texture;

struct Coords {
    int x, y;
};

__host__ __device__ float4 __uchar4_to_float4(uchar4 p) {
    return make_float4(p.x, p.y, p.z, p.w);
}

__host__ __device__ uchar4 __float4_to_uchar4(float4 p) {
    return make_uchar4(p.x, p.y, p.z, p.w);
}

__host__ __device__ float4 __float4_add(float4 u, float4 v) {
    u.x += v.x;
    u.y += v.y;
    u.z += v.z;
    u.w += v.w;
    return u;
}

__host__ __device__ float4 __float4_sub(float4 u, float4 v) {
    u.x -= v.x;
    u.y -= v.y;
    u.z -= v.z;
    u.w -= v.w;
    return u;
}

__host__ __device__ float __float4_dot(float4 u, float4 v) {
    float sum = 0;
    sum += u.x * v.x;
    sum += u.y * v.y;
    sum += u.z * v.z;
    sum += u.w * v.w;
    return sum;
}

__host__ __device__ float4 __float4_div(float4 u, float v) {
    u.x /= v;
    u.y /= v;
    u.z /= v;
    u.w /= v;
    return u;
}

template <size_t NBlocks = 256, size_t NThreads = 256>
void __initialize_averages(const image::Image& image,
                           const std::vector<std::vector<Coords>>& classes) {
    std::vector<float4> avg(classes.size(), make_float4(0, 0, 0, 0));
    for (size_t i = 0; i < classes.size(); ++i) {
        for (auto& p : classes[i]) {
            float4 ps = __uchar4_to_float4(image.data[p.y * image.width + p.x]);
            avg[i] = __float4_add(avg[i], ps);
        }
        avg[i] = __float4_div(avg[i], classes[i].size());
    }
    CHECK_CALL_ERRORS(cudaMemcpyToSymbol(averages, avg.data(),
                                         sizeof(float4) * classes.size(), 0,
                                         cudaMemcpyHostToDevice));
}

void __initialize_texture(cudaArray** array, const image::Image& image) {
    // texture binding
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
    CHECK_CALL_ERRORS(
        cudaMallocArray(array, &channel_desc, image.width, image.height));
    CHECK_CALL_ERRORS(cudaMemcpyToArray(*array, 0, 0, image.data.data(),
                                        sizeof(uchar4) * image.data.size(),
                                        cudaMemcpyHostToDevice));
    CHECK_CALL_ERRORS(
        cudaBindTextureToArray(source_texture, *array, channel_desc));
}

void __free_texture(cudaArray* array) {
    CHECK_CALL_ERRORS(cudaUnbindTexture(source_texture));
    CHECK_CALL_ERRORS(cudaFreeArray(array));
}

__device__ char __classify_pixel(int y, int x, int nclasses) {
    // initial class
    char nclass = -1;
    float max;

    float4 pixel = __uchar4_to_float4(tex2D(source_texture, x, y));
    for (char i = 0; i < nclasses; ++i) {
        float val = -1. * __float4_dot(__float4_sub(pixel, averages[i]),
                                       __float4_sub(pixel, averages[i]));
        if ((nclass == -1) || (val > max)) {
            nclass = i;
            max = val;
        }
    }

    return nclass;
}

__global__ void __classify(uchar4* data, int width, int height, int nclasses) {
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    for (int i = id_y; i < height; i += offset_y) {
        for (int j = id_x; j < width; j += offset_x) {
            data[i * width + j].w = __classify_pixel(i, j, nclasses);
        }
    }
}

template <size_t NBlocks = 256, size_t NThreads = 256>
void MinimumDistance(image::Image& image,
                     const std::vector<std::vector<Coords>>& classes) {
    if (classes.size() > MaxClasses) {
        FATAL("classes limit exceeded");
        return;
    }
    if (classes.size() == 0) {
        return;
    }
    cudaArray* texture_array;
    __initialize_texture(&texture_array, image);
    // pre-calulate averages to store them in constant memory
    __initialize_averages<NBlocks, NThreads>(image, classes);

    gpu::Vector<uchar4> gpu_data(image.data);
    // classify
    __classify<<<NBlocks, NThreads>>>(gpu_data.Data(), image.width,
                                      image.height, classes.size());
    CHECK_KERNEL_ERRORS();

    // copy values back
    gpu_data.Populate(image.data);

    __free_texture(texture_array);
    return;
}
}  // namespace classification

#endif