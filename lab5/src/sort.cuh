#ifndef SORT_CUH
#define SORT_CUH

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <tuple>
#include <type_traits>
#include <vector>
// TODO: remove it!
#include <iostream>

#include "errors.cuh"
#include "vector.cuh"

namespace sort {
template <typename T>
__host__ __device__ T __min(T a, T b) {
    if (a < b) return a;
    return b;
}

template <typename T>
__host__ __device__ T __max(T a, T b) {
    if (a > b) return a;
    return b;
}

template <typename Float>
class __BucketIndexer {
    size_t nbuckets_;
    Float bucket_index_begin_, bucket_index_end_;

public:
    __BucketIndexer(size_t nbuckets, Float min_element, Float max_element)
        : nbuckets_(nbuckets),
          bucket_index_begin_(min_element),
          bucket_index_end_(max_element){};

    __host__ __device__ size_t operator()(Float value_to_index) const {
        Float bucket_width =
            (this->bucket_index_end_ - this->bucket_index_begin_) /
            this->nbuckets_;
        size_t idx = static_cast<size_t>(
            (value_to_index - this->bucket_index_begin_) / bucket_width);
        return __max(size_t{0}, __min(this->nbuckets_ - 1, idx));
    }

    __host__ __device__ size_t nbuckets() const { return this->nbuckets_; }
};

template <typename Float>
struct __DataLimits {
    Float min_element, max_element;
};

template <typename Float, size_t NBlocks, size_t NThreads>
__host__ __DataLimits<Float> __find_data_limits(
    const gpu::Vector<Float>& data) {
    thrust::device_ptr<const Float> data_ptr =
        thrust::device_pointer_cast(data.Data());
    auto min_max = thrust::minmax_element(thrust::device, data_ptr,
                                          data_ptr + data.Size());
    return {*min_max.first, *min_max.second};
}

template <typename Float, size_t NBlocks, size_t NThreads>
__host__ __BucketIndexer<Float> __construct_bucket_indexer(
    const gpu::Vector<Float>& data, size_t nbuckets) {
    __DataLimits<Float> data_limits =
        __find_data_limits<Float, NBlocks, NThreads>(data);
    return __BucketIndexer<Float>(nbuckets, data_limits.min_element,
                                  data_limits.max_element);
}

template <typename Float>
__global__ void __bucket_histogram_kernel(
    size_t* histogram, const Float* data, size_t data_size,
    __BucketIndexer<Float> bucket_indexer) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t offset = blockDim.x * gridDim.x;
    for (size_t it = idx; it < data_size; it += offset) {
        size_t histogram_idx = bucket_indexer(data[it]);
        using ull = unsigned long long;
        atomicAdd(reinterpret_cast<ull*>(&histogram[histogram_idx]), ull{1});
    }
}

template <typename Float, size_t NBlocks, size_t NThreads>
__host__ gpu::Vector<size_t> __bucket_histogram(
    const gpu::Vector<Float>& data,
    const __BucketIndexer<Float>& bucket_indexer) {
    gpu::Vector<size_t> histogram = gpu::MakeVector<size_t, NBlocks, NThreads>(
        bucket_indexer.nbuckets(), size_t{});
    __bucket_histogram_kernel<Float><<<NBlocks, NThreads>>>(
        histogram.Data(), data.Data(), data.Size(), bucket_indexer);
    CHECK_KERNEL_ERRORS();
    return histogram;
}

template <typename T>
__global__ void __blelloch_scan(T* data, size_t data_size) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t offset = blockDim.x * gridDim.x;

    // extern __shared__ T shared[];

    using ll = long long;
    ll layer = 1;
    for (; layer < data_size; layer <<= 1) {
        __syncthreads();
        for (ll j = data_size - 1 - 2 * layer * idx; j - layer >= 0;
             j -= 2 * layer * offset) {
            data[j] += data[j - layer];
        }
    }
    __syncthreads();
    if (idx == 0) {
        data[data_size - 1] = 0;
    }
    for (; layer > 0; layer >>= 1) {
        __syncthreads();
        for (ll j = data_size - 1 - 2 * layer * idx; j - layer >= 0;
             j -= 2 * layer * offset) {
            T t = data[j];
            data[j] += data[j - layer];
            data[j - layer] = t;
        }
    }
}

template <typename T, size_t NBlocks, size_t NThreads>
__host__ void __exclusive_prefix_sum(gpu::Vector<T>& data) {
    if (data.Size() != 0) {
        __blelloch_scan<T><<<NBlocks, NThreads>>>(data.Data(), data.Size());
        CHECK_KERNEL_ERRORS();
    }
}

template <typename Float, size_t NBlocks, size_t NThreads>
__host__ gpu::Vector<size_t> __bucket_offsets(
    const gpu::Vector<Float>& data,
    const __BucketIndexer<Float>& bucket_indexer) {
    gpu::Vector<size_t> bucket_histogram =
        __bucket_histogram<Float, NBlocks, NThreads>(data, bucket_indexer);
    __exclusive_prefix_sum<size_t, NBlocks, NThreads>(bucket_histogram);
    return bucket_histogram;
}

template <typename Float, size_t NBlocks = 256, size_t NThreads = 256>
void BucketSort(std::vector<Float>& data) {
    static_assert(std::is_floating_point<Float>::value,
                  "only floating-point types allowed");
    size_t size = data.size();
    if (size == 0) {
        return;
    }
    gpu::Vector<Float> gpu_data(data);
    __BucketIndexer<Float> bucket_indexer =
        __construct_bucket_indexer<Float, NBlocks, NThreads>(gpu_data, size);
    gpu::Vector<size_t> histogram =
        __bucket_histogram<Float, NBlocks, NThreads>(gpu_data, bucket_indexer);
    {
        // debug
        auto cpu_hist = histogram.Host();
        std::cerr << "histogram " << histogram.Size() << ": ";
        for (auto& it : cpu_hist) {
            std::cerr << it << ' ';
        }
        std::cerr << std::endl;
    }
    gpu::Vector<size_t> offsets =
        __bucket_offsets<Float, NBlocks, NThreads>(gpu_data, bucket_indexer);
    {
        // offsets
        auto cpu_offsets = offsets.Host();
        std::cerr << "offsets " << histogram.Size() << ": ";
        for (auto& it : cpu_offsets) {
            std::cerr << it << ' ';
        }
        std::cerr << std::endl;
    }

    gpu_data.Populate(data);
}
}  // namespace sort

#endif