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
#ifdef DEBUG
    std::cin >> size;
#else
    binary::ReadBinary(std::cin, size);
#endif
    std::vector<float> data(size);
#ifdef DEBUG
    for (auto &it : data) std::cin >> it;
#else
    binary::ReadBinaryArray(std::cin, data.data(), size);
#endif

    sort::BucketSort<float, NBLOCKS, NTHREADS>(data);
#ifdef DEBUG
    std::cerr << size << std::endl;
    std::cerr << "[ ";
    for (auto it : data) std::cerr << it << ' ';
    std::cerr << "]" << std::endl;
#else
    binary::WriteBinaryArray(std::cout, data.data(), size);
#endif
    return 0;
}