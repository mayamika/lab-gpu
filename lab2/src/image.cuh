#ifndef IMAGE_CUH
#define IMAGE_CUH

#include <vector>

#include "binary.cuh"
#include "errors.cuh"

namespace image {
struct Image {
    int width, height;
    std::vector<uchar4> data;
};

Image ReadImage(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);

    Image image;
    binary::ReadBinary(file, image.width);
    binary::ReadBinary(file, image.height);
    if (image.width == 0 || image.height == 0) {
        FATAL("invalid image parameters");
    }
    image.data = std::vector<uchar4>(image.width * image.height);
    binary::ReadBinaryArray(file, image.data.data(), image.data.size());
    return image;
}

void WriteImage(const Image& image, const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);

    binary::WriteBinary(file, image.width);
    binary::WriteBinary(file, image.height);
    binary::WriteBinaryArray(file, image.data.data(), image.data.size());
}
}  // namespace image

#endif