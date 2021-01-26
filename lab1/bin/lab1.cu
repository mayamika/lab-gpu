#include <iostream>
#include <vector>

#include "vector.cuh"

int main() {
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
        ElementwiseMin(gpu::Vector<float>(lhs), gpu::Vector<float>(rhs)).host();

    std::cout.precision(10);
    for (auto& it : mins) {
        std::cout << std::fixed << std::scientific << it << ' ';
    }
    std::cout << '\n';
    return 0;
}