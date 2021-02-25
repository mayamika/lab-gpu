#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include "errors.cuh"
#include "vector.cuh"

namespace matrix {
class Matrix {
    int width_, height_;
    std::vector<double> data_;

public:
    Matrix(int width, int height)
        : width_(width), height_(height), data_(width_ * height_){};

    int Width() const { return this->width_; }
    int Height() const { return this->height_; }

    __host__ __device__ double* Data() { return this->data_.data(); }
    __host__ __device__ const double* Data() const {
        return this->data_.data();
    }

    friend std::istream& operator>>(std::istream&, Matrix&);
    friend std::ostream& operator<<(std::ostream&, const Matrix&);
};

std::istream& operator>>(std::istream& is, Matrix& m) {
    for (int i = 0; i < m.height_; ++i) {
        for (int j = 0; j < m.width_; ++j) {
            is >> m.data_[j * m.height_ + i];
        }
    }
    return is;
}

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    for (int i = 0; i < m.height_; ++i) {
        for (int j = 0; j < m.width_; ++j)
            os << m.data_[i * m.width_ + j] << ' ';
        if (i != m.height_ - 1) os << '\n';
    }
    return os;
}

__global__ void __swap_columns(double* data, int n, int f, int s) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += offset) {
        double tmp = data[i * n + f];
        data[i * n + f] = data[i * n + s];
        data[i * n + s] = tmp;
    }
}

__global__ void __lu_calculate_coefficients(double* data, int n, int i) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    for (int j = i + 1 + idx; j < n; j += offset) {
        data[i * n + j] /= data[i * n + i];
    }
}

__global__ void __lu_columns_substraction(double* data, int n, int i) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    for (int j = i + 1 + idx; j < n; j += offset_x) {
        double coefficient = data[i * n + j];
        for (int k = i + 1 + idy; k < n; k += offset_y) {
            data[k * n + j] -= coefficient * data[k * n + i];
        }
    }
}

__device__ void __lu_solve_row(double* destination_data, double* source_data,
                               int n, int row) {
    for (int j = 0; j < n; ++j) {
        double sum = 0;
        for (int k = 0; k < j; ++k) {
            sum += source_data[k * n + j] * destination_data[k * n + row];
        }
        destination_data[j * n + row] -= sum;
    }

    for (int j = n - 1; j >= 0; --j) {
        double sum = 0;
        for (int k = j + 1; k < n; ++k) {
            sum += source_data[k * n + j] * destination_data[k * n + row];
        }
        destination_data[j * n + row] =
            (destination_data[j * n + row] - sum) / source_data[j * n + j];
    }
}

__global__ void __lu_solve(double* destination_data, double* source_data,
                           int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += offset)
        __lu_solve_row(destination_data, source_data, n, i);
}

struct __less_abs {
    __device__ double operator()(double lhs, double rhs) {
        return fabs(lhs) < fabs(rhs);
    }
};

double eps = 1e-9;

bool __double_is_zero(double val) { return fabs(val) < eps; }

__global__ void __matrix_put_ones(double* data, int n, int* permutation) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += offset) data[i * n + permutation[i]] = 1.;
}

template <size_t NBlocks = 256, size_t NThreads = 256>
void Inverse(Matrix& matrix) {
    if (matrix.Width() != matrix.Height()) {
        FATAL("source matrix must be quadratic");
        return;
    }
    int n = matrix.Width();
    // copy values to gpu
    gpu::Vector<double> source_data(matrix.Data(), n * n);

    // 2D kernel params
    dim3 grid_dim_2d((int)sqrt(NBlocks), (int)sqrt(NBlocks));
    dim3 block_dim_2d((int)sqrt(NThreads), (int)sqrt(NThreads));

    // LUP decomposition
    std::vector<int> rows_permutation(n);
    for (int i = 0; i < rows_permutation.size(); ++i) rows_permutation[i] = i;
    for (int i = 0; i < n; ++i) {
        thrust::device_ptr<double> row_ptr =
            thrust::device_pointer_cast(source_data.Data() + i * n);
        thrust::device_ptr<double> max_ptr =
            thrust::max_element(row_ptr + i, row_ptr + n, __less_abs{});

        double max_value = *max_ptr;
        if (__double_is_zero(max_value)) {
            FATAL("degenerate source matrix");
            return;
        }

        int max_idx = max_ptr - row_ptr;
        std::swap(rows_permutation[i], rows_permutation[max_idx]);
        if (max_idx != i) {
            __swap_columns<<<NBlocks, NThreads>>>(source_data.Data(), n, i,
                                                  max_idx);
            CHECK_KERNEL_ERRORS();
        }

        // lu coefficients calucation
        __lu_calculate_coefficients<<<NBlocks, NThreads>>>(source_data.Data(),
                                                           n, i);
        // lu substraction
        __lu_columns_substraction<<<grid_dim_2d, block_dim_2d>>>(
            source_data.Data(), n, i);
        CHECK_KERNEL_ERRORS();
    }

    // inverse matrix
    gpu::Vector<double> destination_data =
        gpu::MakeVector<double, NBlocks, NThreads>(n * n);
    // initialize with rearranged I matrix
    gpu::Vector<int> gpu_rows_permutation(rows_permutation);
    __matrix_put_ones<<<NBlocks, NThreads>>>(destination_data.Data(), n,
                                             gpu_rows_permutation.Data());
    CHECK_KERNEL_ERRORS();
    __lu_solve<<<NBlocks, NThreads>>>(destination_data.Data(),
                                      source_data.Data(), n);
    CHECK_KERNEL_ERRORS();

    // copy values back
    CHECK_CALL_ERRORS(cudaMemcpy(matrix.Data(), destination_data.Data(),
                                 n * n * sizeof(double),
                                 cudaMemcpyDeviceToHost));

    return;
}
}  // namespace matrix

#endif