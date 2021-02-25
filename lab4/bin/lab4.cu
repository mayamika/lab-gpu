#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "errors.cuh"
#include "matrix.cuh"
#include "signals.cuh"

int main() {
    std::ios::sync_with_stdio(false);
    signals::HandleSignals();

    int size;
    std::cin >> size;

    matrix::Matrix matrix(size, size);
    std::cin >> matrix;
    matrix::Inverse(matrix);

    std::cout.precision(10);
    std::cout << std::scientific << matrix << std::endl;

    return 0;
}
