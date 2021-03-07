#include <iostream>
#include <vector>

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

    int size;
    std::cin >> size;
    if (size == 0) {
        return 0;
    }

    std::vector<float> lhs(size), rhs(size);
    for (auto& it : lhs) {
        std::cin >> it;
    }
    for (auto& it : rhs) {
        std::cin >> it;
    }

    std::vector<float> mins =
        ElementwiseMin(gpu::Vector<float, NBLOCKS, NTHREADS>(lhs),
                       gpu::Vector<float, NBLOCKS, NTHREADS>(rhs))
            .Host();

    std::cout.precision(10);
    std::cout << std::fixed << std::scientific;
    for (auto& it : mins) {
        std::cout << it << ' ';
    }
    std::cout << '\n';

    return 0;
}