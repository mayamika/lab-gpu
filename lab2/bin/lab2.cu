#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "errors.cuh"
#include "signals.cuh"
#include "vector.cuh"

struct Image {
    int width, height;
    std::vector<uchar4> data;
};

texture<uchar4, 2, cudaReadModeElementType> image_texture;
Image read_image(const std::string&);
void write_image(const Image&, const std::string&);
Image apply_blur(const Image&, int);
template <typename T>
void read_binary(std::ifstream& file, T& data) {
    file.read(static_cast<char*>(static_cast<void*>(&data)), sizeof(data));
}
template <typename T>
void write_binary(std::ofstream& file, const T& data) {
    file.write(static_cast<const char*>(static_cast<const void*>(&data)),
               sizeof(data));
}

int main() {
    std::ios::sync_with_stdio(false);
    gpu::handle_signals();

    std::string input_path, output_path;
    std::cin >> input_path >> output_path;
    int radius;
    std::cin >> radius;

    Image image = read_image(input_path);
    image = apply_blur(image, radius);
    write_image(image, output_path);

    return 0;
}

Image read_image(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);

    Image image;
    read_binary<int>(file, image.width);
    read_binary<int>(file, image.height);
    if (image.width == 0 || image.height == 0) {
        FATAL("invalid image parameters");
    }

    image.data = std::vector<uchar4>(image.width * image.height);
    file.read(static_cast<char*>(static_cast<void*>(image.data.data())),
              sizeof(uchar4) * image.data.size());
    return image;
}

void write_image(const Image& image, const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);

    write_binary<int>(file, image.width);
    write_binary<int>(file, image.height);
    file.write(
        static_cast<const char*>(static_cast<const void*>(image.data.data())),
        sizeof(uchar4) * image.data.size());
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

#define PI acosf(-1)

__device__ uchar4 horizontal_gaussian_kernel(int y, int x, int radius) {
    float r = 0, g = 0, b = 0, w = 0;
    for (int i = -radius; i <= radius; ++i) {
        uchar4 p = tex2D(image_texture, x + i, y);
        float mult = expf((-1. * i * i) / (2. * radius * radius));
        // printf("mult: %f\n", mult);
        r += p.x * mult;
        g += p.y * mult;
        b += p.z * mult;
        w += p.w * mult;
    }
    // printf("bef: %f %f %f %f\n", r, g, b, w);
    float den = (radius * sqrtf(2 * PI));
    r /= den;
    g /= den;
    b /= den;
    w /= den;
    // printf("%f %f %f %f\n", r, g, b, w);
    return make_uchar4((unsigned char)roundf(r), (unsigned char)roundf(g),
                       (unsigned char)roundf(b), (unsigned char)roundf(w));
}

__device__ uchar4 vertical_gaussian_kernel(uchar4* data, int width, int height,
                                           int y, int x, int radius) {
    float r = 0, g = 0, b = 0, w = 0;
    for (int i = -radius; i <= radius; ++i) {
        int from_y = max(min(height - 1, y + i), 0);
        uchar4 p = data[from_y * width + x];
        // printf("%f %f %f %d\n", p.x, p.y, p.z, p.w);
        float mult = expf((-1. * i * i) / (2. * radius * radius));
        r += p.x * mult;
        g += p.y * mult;
        b += p.z * mult;
        w += p.w * mult;
    }
    float den = (radius * sqrtf(2 * PI));
    r /= den;
    g /= den;
    b /= den;
    w /= den;
    // printf("%f %f %f %f\n", r, g, b, w);
    return make_uchar4((unsigned char)roundf(r), (unsigned char)roundf(g),
                       (unsigned char)roundf(b), (unsigned char)roundf(w));
}

__global__ void horizontal_gaussian_blur(uchar4* data, int width, int height,
                                         int radius) {
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    for (int i = id_y; i < height; i += offset_y) {
        for (int j = id_x; j < width; j += offset_x) {
            data[i * width + j] = horizontal_gaussian_kernel(i, j, radius);
        }
    }
}

__global__ void vertical_gaussian_blur(uchar4* data, uchar4* first_results,
                                       int width, int height, int radius) {
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    for (int i = id_y; i < height; i += offset_y) {
        for (int j = id_x; j < width; j += offset_x) {
            data[i * width + j] = vertical_gaussian_kernel(
                first_results, width, height, i, j, radius);
        }
    }
}

const size_t NBlocks = 256, NThreads = 256;

Image apply_blur(const Image& source_image, int radius) {
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    cudaArray* texture_array;
    CHECK_CALL_ERRORS(cudaMallocArray(&texture_array, &desc, source_image.width,
                                      source_image.height));
    CHECK_CALL_ERRORS(cudaMemcpyToArray(
        texture_array, 0, 0, source_image.data.data(),
        sizeof(uchar4) * source_image.data.size(), cudaMemcpyHostToDevice));

    image_texture.addressMode[0] = cudaAddressModeClamp;
    image_texture.addressMode[1] = cudaAddressModeClamp;
    image_texture.channelDesc = desc;
    image_texture.filterMode = cudaFilterModePoint;
    image_texture.normalized = false;

    CHECK_CALL_ERRORS(
        cudaBindTextureToArray(image_texture, texture_array, desc));
    gpu::Vector<uchar4, NBlocks, NThreads> result_device(
        source_image.data.size());

    dim3 grid_dim((int)sqrt(NBlocks), (int)sqrt(NBlocks));
    dim3 block_dim((int)sqrt(NThreads), (int)sqrt(NThreads));

    horizontal_gaussian_blur<<<grid_dim, block_dim>>>(
        result_device.Data(), source_image.width, source_image.height, radius);
    CHECK_KERNEL_ERRORS();
    gpu::Vector<uchar4> first_results_copy(result_device);
    vertical_gaussian_blur<<<grid_dim, block_dim>>>(
        result_device.Data(), first_results_copy.Data(), source_image.width,
        source_image.height, radius);
    CHECK_KERNEL_ERRORS();

    CHECK_CALL_ERRORS(cudaUnbindTexture(image_texture));
    CHECK_CALL_ERRORS(cudaFreeArray(texture_array));
    return {source_image.width, source_image.height, result_device.Host()};
}