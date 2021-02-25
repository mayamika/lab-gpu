#ifndef IMAGE_CUH
#define IMAGE_CUH

#include <fstream>
#include <iostream>
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
    if (!file) {
        FATAL("can't open input file");
        return {};
    }
    Image image;
    binary::ReadBinary(file, image.width);
    binary::ReadBinary(file, image.height);
    if ((image.width == 0) || (image.height == 0)) {
        FATAL("invalid image parameters");
        return {};
    }
    image.data = std::vector<uchar4>(image.width * image.height,
                                     make_uchar4(0, 0, 0, 0));
    binary::ReadBinaryArray(file, image.data.data(), image.data.size());
    return image;
}

void WriteImage(const Image& image, const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        FATAL("can't open output file");
        return;
    }
    binary::WriteBinary(file, image.width);
    binary::WriteBinary(file, image.height);
    binary::WriteBinaryArray(file, image.data.data(), image.data.size());
}

void WriteStderr(const Image& image) {
    std::cerr << image.width << ' ' << image.height << '\n';
    for (auto it : image.data)
        std::cerr << (int)it.x << ' ' << (int)it.y << ' ' << (int)it.z << ' '
                  << (int)it.w << '\n';
}
}  // namespace image

#endif