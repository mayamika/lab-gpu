#ifndef VECTOR_CUH
#define VECTOR_CUH

#include <cstring>
#include <vector>

#include "errors.cuh"

#ifdef __INTELLISENSE__
#define __global__
#define __device__
#define __host__
#endif

namespace gpu {

template <class Type>
__global__ void __memset(Type* data, size_t size, Type value) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t offset = blockDim.x * gridDim.x;
    for (size_t it = idx; it < size; it += offset) {
        data[it] = value;
    }
}

template <class Type>
__global__ void __elementwise_min(Type* dst, size_t size, const Type* lhs,
                                  const Type* rhs) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t offset = blockDim.x * gridDim.x;
    for (size_t it = idx; it < size; it += offset) {
        dst[it] = fmin(lhs[it], rhs[it]);
    }
}

template <class Type, size_t NBlocks = 256, size_t NThreads = 256>
class Vector {
private:
    size_t _size;
    Type* _data;

public:
    Vector() : _size(0), _data(nullptr) {}

    Vector(size_t size, Type initializer = Type())
        : _size(size), _data(nullptr) {
        CHECK_CALL_ERRORS(cudaMalloc(&this->_data, sizeof(Type) * this->_size));
        __memset<<<NBlocks, NThreads>>>(this->_data, this->_size, initializer);
        CHECK_KERNEL_ERRORS();
    }

    Vector(const Vector& src) : _size(src._size), _data(nullptr) {
        CHECK_CALL_ERRORS(cudaMalloc(&this->_data, sizeof(Type) * this->_size));
        CHECK_CALL_ERRORS(cudaMemcpy(this->_data, src._data,
                                     sizeof(Type) * src._size,
                                     cudaMemcpyHostToDevice));
    }

    Vector(Vector&& src) : _size(src._size), _data(src._data) {}

    Vector(const std::vector<Type>& src) : _size(src.size()), _data(nullptr) {
        CHECK_CALL_ERRORS(cudaMalloc(&this->_data, sizeof(Type) * this->_size));
        CHECK_CALL_ERRORS(cudaMemcpy(this->_data, src.data(),
                                     sizeof(Type) * src.size(),
                                     cudaMemcpyHostToDevice));
    };

    size_t size() const { return this->_size; }

    void clear() {
        if (this->_data != nullptr) {
            CHECK_CALL_ERRORS(cudaFree(this->_data));
        }
        this->_size = 0;
        this->_data = nullptr;
    }

    void populate(std::vector<Type>& dst) const {
        dst.resize(this->_size);
        CHECK_CALL_ERRORS(cudaMemcpy(dst.data(), this->_data,
                                     sizeof(Type) * this->_size,
                                     cudaMemcpyDeviceToHost));
    }

    std::vector<Type> host() const {
        std::vector<Type> result(this->_size);
        CHECK_CALL_ERRORS(cudaMemcpy(result.data(), this->_data,
                                     sizeof(Type) * this->_size,
                                     cudaMemcpyDeviceToHost));
        return result;
    }

    ~Vector() { this->clear(); };

    const Vector& operator=(const Vector& src) {
        *this = Vector(src);
        return *this;
    }

    const Vector& operator=(Vector&& src) {
        *this = Vector(std::move(src));
        return *this;
    }

    friend Vector ElementwiseMin(const Vector& lhs, const Vector& rhs) {
        if (lhs._size != rhs._size) {
            FATAL("lhs.size() != rhs.size()");
        }
        size_t size = lhs._size;
        Vector result(size);
        __elementwise_min<<<NBlocks, NThreads>>>(result._data, size, lhs._data,
                                                 rhs._data);
        CHECK_KERNEL_ERRORS();
        return result;
    }
};
}  // namespace gpu

#endif