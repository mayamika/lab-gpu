#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "blur.cuh"
#include "errors.cuh"
#include "image.cuh"
#include "signals.cuh"
#include "vector.cuh"

#ifndef NBLOCKS
#define NBLOCKS 256
#endif
#ifndef NTHREADS
#define NTHREADS 256
#endif

int main() {
#ifdef BENCHMARK
    std::cerr << NBLOCKS << ' ' << NTHREADS << '\n';
#endif
    std::ios::sync_with_stdio(false);
    signals::HandleSignals();

    std::string input_path, output_path;
    std::cin >> input_path >> output_path;
    int radius;
    std::cin >> radius;

    image::Image image = image::ReadImage(input_path);
    blur::ApplyGaussianBlur(image, radius);
    image::WriteImage(image, output_path);
    return 0;
}
