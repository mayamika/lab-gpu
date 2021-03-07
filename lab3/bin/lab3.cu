#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "blur.cuh"
#include "classification.cuh"
#include "errors.cuh"
#include "image.cuh"
#include "signals.cuh"

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

    int nclasses;
    std::cin >> nclasses;
    std::vector<std::vector<classification::Coords>> classes(nclasses);
    for (int i = 0; i < nclasses; ++i) {
        int np;
        std::cin >> np;

        std::vector<classification::Coords> coords(np);
        for (auto& it : coords) {
            std::cin >> it.x >> it.y;
        }
        classes[i] = coords;
    }

    image::Image image = image::ReadImage(input_path);
    classification::MinimumDistance(image, classes);
    image::WriteImage(image, output_path);
    return 0;
}
