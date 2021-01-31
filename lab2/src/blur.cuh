#ifndef BLUR_CUH
#define BLUR_CUH

#include <cmath>

#include "errors.cuh"
#include "image.cuh"
#include "vector.cuh"

namespace blur {
template <typename T>
__device__ T abs(T a) {
    if (a < 0) return -a;
    return a;
}

template <typename T>
__device__ T min(T a, T b) {
    if (a < b) return a;
    return b;
}

template <typename T>
__device__ T max(T a, T b) {
    if (a > b) return a;
    return b;
}

float gaussian_fuction(float x, float r) {
    return 1.0 / sqrt(2.0 * M_PI * r * r) * exp(-x * x / (2.0 * r * r));
}

std::vector<float> get_gaussian_weights(int r) {
    std::vector<float> filter(r + 1);
    float sum = 0;
    for (int i = 0; i <= r; ++i) {
        filter[i] = gaussian_fuction(i, r);
        sum += filter[i];
    }
    sum = 2 * (sum - filter[0]) + sum;
    for (int i = 0; i <= r; ++i) {
        filter[i] /= (sum);
    }
    return filter;
}

using _CudaTexture = texture<uchar4, 2, cudaReadModeElementType>;
_CudaTexture global_texture;

struct _Texture {
    cudaArray* array;

    _Texture(const image::Image& image) {
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
        CHECK_CALL_ERRORS(cudaMallocArray(&(this->array), &channel_desc,
                                          image.width, image.height));
        CHECK_CALL_ERRORS(cudaMemcpyToArray(
            this->array, 0, 0, image.data.data(),
            sizeof(uchar4) * image.data.size(), cudaMemcpyHostToDevice));
        CHECK_CALL_ERRORS(
            cudaBindTextureToArray(global_texture, this->array, channel_desc));
    }

    _Texture(const _Texture&) = delete;
    const _Texture& operator=(const _Texture&) = delete;

    ~_Texture() {
        CHECK_CALL_ERRORS(cudaUnbindTexture(global_texture));
        CHECK_CALL_ERRORS(cudaFreeArray(this->array));
    }
};

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

__device__ float4 uchar4_to_float4(uchar4 p) {
    return make_float4(p.x, p.y, p.z, p.w);
}

__device__ uchar4 gaussian_blur_kernel(int y, int x, int radius,
                                       const float* weights, bool vertical) {
    float4 result = make_float4(0, 0, 0, 0);
    for (int i = -radius; i <= radius; ++i) {
        float4 p = uchar4_to_float4(tex2D(
            global_texture, (vertical) ? x : x + i, (vertical) ? y + i : y));
        // printf("%f %f %f %d\n", p.x, p.y, p.z, p.w);
        result = float4_add(result, float4_multiply(p, weights[abs(i)]));
    }
    // printf("%f %f %f %f\n", r, g, b, w);
    return make_uchar4(roundf(result.x), roundf(result.y), roundf(result.z),
                       roundf(result.w));
}

__global__ void gaussian_blur(uchar4* data, int width, int height, int radius,
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
image::Image ApplyGaussianBlur(const image::Image& source_image, int radius) {
    _Texture texture(source_image);

    dim3 grid_dim((int)sqrt(NBlocks), (int)sqrt(NBlocks));
    dim3 block_dim((int)sqrt(NThreads), (int)sqrt(NThreads));

    gpu::Vector<float, NBlocks, NThreads> weights(get_gaussian_weights(radius));
    // horizontal
    gpu::Vector<uchar4, NBlocks, NThreads> first_results(
        source_image.data.size(), make_uchar4(0, 0, 0, 0));
    gaussian_blur<<<grid_dim, block_dim>>>(
        first_results.Data(), source_image.width, source_image.height, radius,
        weights.Data(), false);
    CHECK_KERNEL_ERRORS();

    // copy results
    CHECK_CALL_ERRORS(cudaMemcpyToArray(
        texture.array, 0, 0, first_results.Data(),
        sizeof(uchar4) * first_results.Size(), cudaMemcpyDeviceToDevice));

    // vertical
    gpu::Vector<uchar4, NBlocks, NThreads> second_results(
        source_image.data.size(), make_uchar4(0, 0, 0, 0));
    gaussian_blur<<<grid_dim, block_dim>>>(
        second_results.Data(), source_image.width, source_image.height, radius,
        weights.Data(), true);
    CHECK_KERNEL_ERRORS();

    return {source_image.width, source_image.height, second_results.Host()};
}
}  // namespace blur

#endif