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

int main() {
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
