#ifndef VECTOR_CUH
#define VECTOR_CUH

#include <cstring>
#include <vector>

#include "errors.cuh"

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

template <class Type>
class Vector {
private:
    size_t size_;
    Type* data_;

public:
    Vector() : size_(0), data_(nullptr) {}

    Vector(const Vector& src) : size_(src.size_), data_(nullptr) {
        CHECK_CALL_ERRORS(cudaMalloc(&this->data_, sizeof(Type) * this->size_));
        CHECK_CALL_ERRORS(cudaMemcpy(this->data_, src.data_,
                                     sizeof(Type) * src.size_,
                                     cudaMemcpyDeviceToDevice));
    }

    Vector(Vector&& src) : size_(src.size_), data_(src.data_) {}

    Vector(const std::vector<Type>& src) : size_(src.size()), data_(nullptr) {
        CHECK_CALL_ERRORS(cudaMalloc(&this->data_, sizeof(Type) * this->size_));
        CHECK_CALL_ERRORS(cudaMemcpy(this->data_, src.data(),
                                     sizeof(Type) * src.size(),
                                     cudaMemcpyHostToDevice));
    };

    Vector(const Type* data, size_t size) : size_(size), data_(nullptr) {
        CHECK_CALL_ERRORS(cudaMalloc(&this->data_, sizeof(Type) * this->size_));
        CHECK_CALL_ERRORS(cudaMemcpy(this->data_, data,
                                     sizeof(Type) * this->size_,
                                     cudaMemcpyHostToDevice));
    };

    __host__ __device__ size_t Size() const { return this->size_; }

    __host__ __device__ const Type* Data() const { return this->data_; }

    __host__ __device__ Type* Data() { return this->data_; }

    void Clear() {
        if (this->data_ != nullptr) {
            CHECK_CALL_ERRORS(cudaFree(this->data_));
        }
        this->size_ = 0;
        this->data_ = nullptr;
    }

    void Populate(std::vector<Type>& dst) const {
        dst.resize(this->size_);
        CHECK_CALL_ERRORS(cudaMemcpy(dst.data(), this->data_,
                                     sizeof(Type) * this->size_,
                                     cudaMemcpyDeviceToHost));
    }

    std::vector<Type> Host() const {
        std::vector<Type> result(this->size_);
        CHECK_CALL_ERRORS(cudaMemcpy(result.data(), this->data_,
                                     sizeof(Type) * this->size_,
                                     cudaMemcpyDeviceToHost));
        return result;
    }

    ~Vector() { this->Clear(); };

    const Vector& operator=(const Vector& src) {
        *this = Vector(src);
        return *this;
    }

    const Vector& operator=(Vector&& src) {
        *this = Vector(std::move(src));
        return *this;
    }

    template <typename CType, size_t NBlocks, size_t NThreads>
    friend Vector<CType> MakeVector(size_t, CType);

    template <size_t NBlocks = 256, size_t NThreads = 256>
    friend Vector ElementwiseMin(const Vector& lhs, const Vector& rhs) {
        if (lhs.size_ != rhs.size_) {
            FATAL("lhs.size() != rhs.size()");
        }
        size_t size = lhs.size_;
        Vector result(size);
        __elementwise_min<<<NBlocks, NThreads>>>(result.data_, size, lhs.data_,
                                                 rhs.data_);
        CHECK_KERNEL_ERRORS();
        return result;
    }
};

template <typename Type, size_t NBlocks = 256, size_t NThreads = 256>
Vector<Type> MakeVector(size_t size, Type initializer = Type()) {
    gpu::Vector<Type> vector;
    vector.size_ = size;
    CHECK_CALL_ERRORS(cudaMalloc(&vector.data_, sizeof(Type) * vector.size_));
    __memset<<<NBlocks, NThreads>>>(vector.data_, vector.size_, initializer);
    CHECK_KERNEL_ERRORS();
    return vector;
}

}  // namespace gpu

#endif