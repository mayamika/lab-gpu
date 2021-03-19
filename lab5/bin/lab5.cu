#include <iostream>
#include <vector>

#include "binary.cuh"
#include "signals.cuh"
#include "sort.cuh"

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

    uint32_t size;
    binary::ReadBinary(std::cin, size);
    std::vector<float> data(size);
    binary::ReadBinaryArray(std::cin, data.data(), size);
    std::cerr << '[';
    for (auto it : data) {
        std::cerr << it << ' ';
    }
    std::cerr << ']' << std::endl;

    sort::BucketSort(data);
    // binary::WriteBinaryArray(std::cout, data.data(), size);

    return 0;
}