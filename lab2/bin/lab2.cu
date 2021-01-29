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
    gpu::Vector<uchar4, 256, 256> result_device(source_image.data.size());

    CHECK_CALL_ERRORS(cudaUnbindTexture(image_texture));
    CHECK_CALL_ERRORS(cudaFreeArray(texture_array));
    return {source_image.width, source_image.height, result_device.Host()};
}