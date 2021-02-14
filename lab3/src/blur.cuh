#ifndef BLUR_CUH
#define BLUR_CUH

#include <cmath>

#include "errors.cuh"
#include "image.cuh"
#include "vector.cuh"

namespace blur {
template <typename T>
__host__ __device__ T abs(T a) {
    if (a < 0) return -a;
    return a;
}

__host__ __device__ float gaussian_fuction(float x, float r) {
    return 1.0 / sqrt(2.0 * M_PI * r * r) * exp(-x * x / (2.0 * r * r));
}

std::vector<float> get_gaussian_weights(int r) {
    std::vector<float> filter(r + 1);
    float sum = 0;
    for (int i = 0; i <= r; ++i) {
        filter[i] = gaussian_fuction(i, r);
        sum += filter[i];
    }
    sum = 2 * (sum - filter[0]) + filter[0];
    for (int i = 0; i <= r; ++i) {
        filter[i] /= (sum);
    }
    return filter;
}

__device__ float4 float4_add(float4 u, float v) {
    u.x += v;
    u.y += v;
    u.z += v;
    u.w += v;
    return u;
}

__device__ float4 float4_add(float4 u, float4 v) {
    u.x += v.x;
    u.y += v.y;
    u.z += v.z;
    u.w += v.w;
    return u;
}

__device__ float4 float4_multiply(float4 u, float v) {
    u.x *= v;
    u.y *= v;
    u.z *= v;
    u.w *= v;
    return u;
}

__device__ float4 float4_multiply(float4 u, float4 v) {
    u.x *= v.x;
    u.y *= v.y;
    u.z *= v.z;
    u.w *= v.w;
    return u;
}

__host__ __device__ float4 uchar4_to_float4(uchar4 p) {
    return make_float4(p.x, p.y, p.z, p.w);
}

__host__ __device__ uchar4 float4_to_uchar4(float4 p) {
    return make_uchar4((p.x), (p.y), (p.z), (p.w));
}

using _CudaTexture = texture<float4, 2, cudaReadModeElementType>;
_CudaTexture source_texture, intermediate_results_texture;

__device__ float4 gaussian_blur_kernel(int y, int x, int radius,
                                       const float* weights, bool vertical) {
    float4 result = make_float4(0, 0, 0, 0);
    for (int i = -radius; i <= radius; ++i) {
        float4 p =
            tex2D((vertical) ? intermediate_results_texture : source_texture,
                  (vertical) ? x : x + i, (vertical) ? y + i : y);
        result = float4_add(result, float4_multiply(p, weights[abs(i)]));
    }
    return result;
}

__global__ void gaussian_blur(float4* data, int width, int height, int radius,
                              const float* weights, bool vertical) {
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    for (int i = id_y; i < height; i += offset_y) {
        for (int j = id_x; j < width; j += offset_x) {
            data[i * width + j] =
                gaussian_blur_kernel(i, j, radius, weights, vertical);
        }
    }
}

template <size_t NBlocks = 256, size_t NThreads = 256>
void ApplyGaussianBlur(image::Image& source_image, int radius) {
    if (radius == 0) {
        return;
    }
    if (radius < 0) {
        FATAL("invalid radius")
        return;
    }

    // transform source data to floats
    std::vector<float4> source_data(source_image.data.size());
    for (size_t i = 0; i < source_data.size(); ++i)
        source_data[i] = uchar4_to_float4(source_image.data[i]);

    // texture initialization
    cudaArray* source_array;
    {
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
        CHECK_CALL_ERRORS(cudaMallocArray(&(source_array), &channel_desc,
                                          source_image.width,
                                          source_image.height));
        CHECK_CALL_ERRORS(cudaMemcpyToArray(
            source_array, 0, 0, source_data.data(),
            sizeof(float4) * source_data.size(), cudaMemcpyHostToDevice));
        CHECK_CALL_ERRORS(
            cudaBindTextureToArray(source_texture, source_array, channel_desc));
    }

    // kernel params
    dim3 grid_dim((int)sqrt(NBlocks), (int)sqrt(NBlocks));
    dim3 block_dim((int)sqrt(NThreads), (int)sqrt(NThreads));

    gpu::Vector<float> weights(get_gaussian_weights(radius));
    // horizontal
    gpu::Vector<float4> first_results =
        gpu::MakeVector<float4, NBlocks, NThreads>(source_image.data.size(),
                                                   make_float4(0, 0, 0, 0));
    gaussian_blur<<<grid_dim, block_dim>>>(
        first_results.Data(), source_image.width, source_image.height, radius,
        weights.Data(), false);
    CHECK_KERNEL_ERRORS();

    // source texture cleanup
    CHECK_CALL_ERRORS(cudaUnbindTexture(source_texture));
    CHECK_CALL_ERRORS(cudaFreeArray(source_array));

    // intermediate results texture initialization
    cudaArray* intermediate_results_array;
    {
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
        CHECK_CALL_ERRORS(cudaMallocArray(&(intermediate_results_array),
                                          &channel_desc, source_image.width,
                                          source_image.height));
        CHECK_CALL_ERRORS(cudaMemcpyToArray(
            intermediate_results_array, 0, 0, first_results.Data(),
            sizeof(float4) * first_results.Size(), cudaMemcpyDeviceToDevice));
        CHECK_CALL_ERRORS(cudaBindTextureToArray(intermediate_results_texture,
                                                 intermediate_results_array,
                                                 channel_desc));
        first_results.Clear();
    }

    // vertical
    gpu::Vector<float4> second_results =
        gpu::MakeVector<float4, NBlocks, NThreads>(source_image.data.size(),
                                                   make_float4(0, 0, 0, 0));
    gaussian_blur<<<grid_dim, block_dim>>>(
        second_results.Data(), source_image.width, source_image.height, radius,
        weights.Data(), true);
    CHECK_KERNEL_ERRORS();

    // intermediate results texture cleanup
    CHECK_CALL_ERRORS(cudaUnbindTexture(intermediate_results_texture));
    CHECK_CALL_ERRORS(cudaFreeArray(intermediate_results_array));

    std::vector<float4> results_float4 = second_results.Host();
    for (size_t i = 0; i < source_image.data.size(); ++i) {
        source_image.data[i] = float4_to_uchar4(results_float4[i]);
    }

    return;
}
}  // namespace blur

#endif