#ifndef SORT_CUH
#define SORT_CUH

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <iostream>
#include <tuple>
#include <type_traits>
#include <vector>

#include "errors.cuh"
#include "vector.cuh"

namespace sort {
template <typename T>
__host__ __device__ T __max(T a, T b) {
    if (a > b) return a;
    return b;
}

template <typename T>
__host__ __device__ T __min(T a, T b) {
    if (a < b) return a;
    return b;
}

template <typename Float>
class __BucketIndexer {
    uint32_t nbuckets_;
    Float bucket_index_begin_, bucket_index_end_;

public:
    __BucketIndexer(uint32_t nbuckets, Float min_element, Float max_element)
        : nbuckets_(nbuckets),
          bucket_index_begin_(min_element),
          bucket_index_end_(max_element){};

    __host__ __device__ uint32_t operator()(Float value_to_index) const {
        Float bucket_width =
            (this->bucket_index_end_ - this->bucket_index_begin_) /
            this->nbuckets_;
        uint32_t idx = static_cast<uint32_t>(
            (value_to_index - this->bucket_index_begin_) / bucket_width);
        return __min(this->nbuckets_ - 1, idx);
    }

    __host__ __device__ uint32_t nbuckets() const { return this->nbuckets_; }
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
__host__ __BucketIndexer<Float> __make_bucket_indexer(
    const gpu::Vector<Float>& data, uint32_t nbuckets) {
    __DataLimits<Float> data_limits =
        __find_data_limits<Float, NBlocks, NThreads>(data);
    return __BucketIndexer<Float>(nbuckets, data_limits.min_element,
                                  data_limits.max_element);
}

template <typename Float>
__global__ void __bucket_histogram_kernel(
    uint32_t* histogram, const Float* data, size_t data_size,
    __BucketIndexer<Float> bucket_indexer) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t offset = blockDim.x * gridDim.x;

    for (size_t it = idx; it < data_size; it += offset) {
        uint32_t histogram_idx = bucket_indexer(data[it]);
        atomicAdd((&histogram[histogram_idx]), uint32_t{1});
    }
}

template <typename Float, size_t NBlocks, size_t NThreads>
__host__ gpu::Vector<uint32_t> __bucket_histogram(
    const gpu::Vector<Float>& data,
    const __BucketIndexer<Float>& bucket_indexer) {
    gpu::Vector<uint32_t> histogram =
        gpu::MakeVector<uint32_t, NBlocks, NThreads>(bucket_indexer.nbuckets(),
                                                     uint32_t{});
    __bucket_histogram_kernel<Float><<<NBlocks, NThreads>>>(
        histogram.Data(), data.Data(), data.Size(), bucket_indexer);
    CHECK_KERNEL_ERRORS();
    return histogram;
}

template <typename T, size_t NThreads>
__global__ void __single_block_blelloch_scan(T* data, int64_t size) {
    int64_t idx = threadIdx.x;
    __shared__ T shared[2 * NThreads];
    {
        int64_t j = size - 2 * idx - 1;
        if (j >= 0) shared[j] = data[j];
        if (j - 1 >= 0) shared[j - 1] = data[j - 1];
    }
    int64_t layer = 1;
    for (; layer < size; layer <<= 1) {
        __syncthreads();
        int64_t j = size - 1 - 2 * layer * idx;
        if (j - layer >= 0) {
            shared[j] += shared[j - layer];
        }
    }
    __syncthreads();
    if (idx == 0) {
        shared[size - 1] = 0;
    }
    for (; layer > 0; layer >>= 1) {
        __syncthreads();
        int64_t j = size - 1 - 2 * layer * idx;
        if (j - layer >= 0) {
            T t = shared[j];
            shared[j] += shared[j - layer];
            shared[j - layer] = t;
        }
    }
    __syncthreads();
    {
        int64_t j = size - 2 * idx - 1;
        if (j >= 0) data[j] = shared[j];
        if (j - 1 >= 0) data[j - 1] = shared[j - 1];
    }
}

template <typename T, size_t NThreads>
__global__ void __partial_blelloch_scan(T* data, int64_t size,
                                        T* partial_sums) {
    int64_t tidx = threadIdx.x;
    int64_t offset = blockDim.x * gridDim.x;
    for (int64_t segment_begin = blockIdx.x * blockDim.x; segment_begin < size;
         segment_begin += offset) {
        int64_t segment_end = __min(segment_begin + blockDim.x, size);
        int64_t segment_size = segment_end - segment_begin;
        __syncthreads();
        __shared__ T shared[2 * NThreads];
        {
            int64_t j = segment_end - 2 * tidx - 1;
            if (j >= segment_begin) shared[j - segment_begin] = data[j];
            if (j - 1 >= segment_begin)
                shared[j - 1 - segment_begin] = data[j - 1];
        }
        int64_t layer = 1;
        for (; layer < segment_size; layer <<= 1) {
            __syncthreads();
            int64_t j = segment_size - 1 - 2 * layer * tidx;
            if (j - layer >= 0) {
                shared[j] += shared[j - layer];
            }
        }
        __syncthreads();
        if (tidx == 0) {
            partial_sums[segment_begin / blockDim.x] = shared[segment_size - 1];
            shared[segment_size - 1] = 0;
        }
        for (; layer > 0; layer >>= 1) {
            __syncthreads();
            int64_t j = segment_size - 1 - 2 * layer * tidx;
            if (j - layer >= 0) {
                T t = shared[j];
                shared[j] += shared[j - layer];
                shared[j - layer] = t;
            }
        }
        __syncthreads();
        {
            int64_t j = segment_end - 2 * tidx - 1;
            if (j >= segment_begin) data[j] = shared[j - segment_begin];
            if (j - 1 >= segment_begin)
                data[j - 1] = shared[j - 1 - segment_begin];
        }
    }
}

template <typename T>
__global__ void __merge_scans(T* data, size_t data_size, T* partial_sums) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t offset = blockDim.x * gridDim.x;

    for (size_t i = idx; i < data_size; i += offset) {
        data[i] += partial_sums[i / blockDim.x];
    }
}

#ifdef DEBUG
template <typename T>
__host__ void __debug_print_data(const gpu::Vector<T>& data) {
    auto cpu = data.Host();
    std::cerr << cpu.size() << std::endl;
    for (auto it : cpu) {
        std::cerr << it << ' ';
    }
    std::cerr << std::endl;
}
#endif

template <typename T, size_t NBlocks, size_t NThreads>
__host__ void __exclusive_prefix_sum(gpu::Vector<T>& data) {
    size_t size = data.Size();
    if (size < 2) {
        return;
    }
    if (size <= 2 * NThreads) {
        __single_block_blelloch_scan<T, NThreads>
            <<<1, NThreads>>>(data.Data(), data.Size());
        CHECK_KERNEL_ERRORS();
        return;
    }
    // partial recursive scan
    gpu::Vector<T> partial_sums = gpu::MakeVector<T, NBlocks, NThreads>(
        (size + NThreads - 1) / NThreads, T{});
    __partial_blelloch_scan<T, NThreads>
        <<<NBlocks, NThreads>>>(data.Data(), data.Size(), partial_sums.Data());
    cudaDeviceSynchronize();
    CHECK_KERNEL_ERRORS();
    __exclusive_prefix_sum<T, NBlocks, NThreads>(partial_sums);
    cudaDeviceSynchronize();
    CHECK_KERNEL_ERRORS();
    __merge_scans<T>
        <<<NBlocks, NThreads>>>(data.Data(), data.Size(), partial_sums.Data());
    CHECK_KERNEL_ERRORS();
}

template <typename Float, size_t NBlocks, size_t NThreads>
__host__ gpu::Vector<uint32_t> __bucket_offsets(
    const gpu::Vector<Float>& data,
    const __BucketIndexer<Float>& bucket_indexer) {
    gpu::Vector<uint32_t> bucket_histogram =
        __bucket_histogram<Float, NBlocks, NThreads>(data, bucket_indexer);
    cudaDeviceSynchronize();
    __exclusive_prefix_sum<uint32_t, NBlocks, NThreads>(bucket_histogram);
    return bucket_histogram;
}

template <typename Float>
__global__ void __bucket_group_kernel(Float* dst_data, const Float* data,
                                      size_t data_size, const uint32_t* offsets,
                                      uint32_t* bucket_sizes,
                                      __BucketIndexer<Float> bucket_indexer) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t offset = blockDim.x * gridDim.x;

    for (size_t it = idx; it < data_size; it += offset) {
        uint32_t bucket_index = bucket_indexer(data[it]);
        uint32_t bucket_size =
            atomicSub(&bucket_sizes[bucket_index], uint32_t{1});
        dst_data[offsets[bucket_index] + (bucket_size - 1)] = data[it];
    }
}

template <typename Float, size_t NBlocks, size_t NThreads>
__host__ gpu::Vector<Float> __bucket_group(
    const gpu::Vector<Float>& data, const gpu::Vector<uint32_t>& offsets,
    const __BucketIndexer<Float>& bucket_indexer) {
    // histogram to store bucket sizes
    gpu::Vector<uint32_t> sizes =
        __bucket_histogram<Float, NBlocks, NThreads>(data, bucket_indexer);
    gpu::Vector<Float> dst_data =
        gpu::MakeVector<Float, NBlocks, NThreads>(data.Size(), Float{});
    cudaDeviceSynchronize();
    __bucket_group_kernel<Float>
        <<<NBlocks, NThreads>>>(dst_data.Data(), data.Data(), data.Size(),
                                offsets.Data(), sizes.Data(), bucket_indexer);
    CHECK_KERNEL_ERRORS();
    return dst_data;
}

template <typename T>
__global__ void __odd_even_sort_kernel(T* data, size_t begin, size_t end,
                                       int i) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t offset = blockDim.x * gridDim.x;

    size_t j = begin + 2 * idx + (!(i & 1));
    for (; j + 1 < end; j += 2 * offset) {
        if (data[j] > data[j + 1]) {
            T t = data[j];
            data[j] = data[j + 1];
            data[j + 1] = t;
        }
    }
}

template <typename T, size_t NThreads>
__global__ void __single_block_odd_even_sort_kernel(T* data, int64_t begin,
                                                    int64_t end) {
    int64_t tidx = threadIdx.x;
    int64_t size = end - begin;
    __shared__ T shared[2 * NThreads];
    {
        int64_t j = begin + 2 * tidx;
        if (j < end) shared[j - begin] = data[j];
        if (j + 1 < end) shared[j - begin + 1] = data[j + 1];
    }
    for (size_t i = 0; i < size; ++i) {
        __syncthreads();
        size_t j = 2 * tidx + (!(i & 1));
        if (j + 1 < size && shared[j] > shared[j + 1]) {
            T t = shared[j];
            shared[j] = shared[j + 1];
            shared[j + 1] = t;
        }
    }
    {
        __syncthreads();
        int64_t j = begin + 2 * tidx;
        if (j < end) data[j] = shared[j - begin];
        if (j + 1 < end) data[j + 1] = shared[j - begin + 1];
    }
}

template <typename T, size_t NBlocks, size_t NThreads>
__host__ void __odd_even_sort(gpu::Vector<T>& data, size_t begin, size_t end) {
    size_t size = end - begin;
    if (size < 2) {
        return;
    }
    if (size <= 2 * NThreads) {
        __single_block_odd_even_sort_kernel<T, NThreads>
            <<<1, NThreads>>>(data.Data(), begin, end);
        CHECK_KERNEL_ERRORS();
        return;
    }
    for (size_t i = begin; i < end; ++i) {
        cudaDeviceSynchronize();
        __odd_even_sort_kernel<T>
            <<<NBlocks, NThreads>>>(data.Data(), begin, end, i);
        CHECK_KERNEL_ERRORS();
    }
}

template <typename T, size_t NThreads>
__global__ void __partial_odd_even_sort_kernel(T* data, int64_t size,
                                               const uint32_t* offsets,
                                               int64_t nbuckets) {
    int64_t tidx = threadIdx.x;
    for (int64_t bucket = blockIdx.x; bucket < nbuckets; bucket += gridDim.x) {
        int64_t segment_begin = offsets[bucket];
        int64_t segment_end =
            (bucket + 1 == nbuckets) ? size : offsets[bucket + 1];
        int64_t segment_size = segment_end - segment_begin;
        __syncthreads();
        __shared__ T shared[2 * NThreads];
        {
            int64_t j = segment_begin + 2 * tidx;
            if (j < segment_end) shared[j - segment_begin] = data[j];
            if (j + 1 < segment_end)
                shared[j - segment_begin + 1] = data[j + 1];
        }
        for (size_t i = 0; i < segment_size; ++i) {
            __syncthreads();
            size_t j = 2 * tidx + (!(i & 1));
            if (j + 1 < segment_size && shared[j] > shared[j + 1]) {
                T t = shared[j];
                shared[j] = shared[j + 1];
                shared[j + 1] = t;
            }
        }
        {
            __syncthreads();
            int64_t j = segment_begin + 2 * tidx;
            if (j < segment_end) data[j] = shared[j - segment_begin];
            if (j + 1 < segment_end)
                data[j + 1] = shared[j - segment_begin + 1];
        }
    }
}

template <typename T, size_t NBlocks, size_t NThreads>
__host__ void __partial_odd_even_sort(gpu::Vector<T>& data,
                                      const gpu::Vector<uint32_t>& offsets) {
    __partial_odd_even_sort_kernel<T, NThreads><<<NBlocks, NThreads>>>(
        data.Data(), data.Size(), offsets.Data(), offsets.Size());
    CHECK_KERNEL_ERRORS();
}

template <typename Float, size_t NBlocks = 256, size_t NThreads = 256>
void BucketSort(std::vector<Float>& data) {
    static_assert(std::is_floating_point<Float>::value,
                  "only floating-point types allowed");
    size_t size = data.size();
    if (size < 2) {
        return;
    }
    gpu::Vector<Float> gpu_data(data);
    __BucketIndexer<Float> bucket_indexer =
        __make_bucket_indexer<Float, NBlocks, NThreads>(gpu_data, size);
    gpu::Vector<uint32_t> offsets =
        __bucket_offsets<Float, NBlocks, NThreads>(gpu_data, bucket_indexer);
    cudaDeviceSynchronize();
    gpu::Vector<Float> grouped_data = __bucket_group<Float, NBlocks, NThreads>(
        gpu_data, offsets, bucket_indexer);
    cudaDeviceSynchronize();
    __partial_odd_even_sort<Float, NBlocks, NThreads>(grouped_data, offsets);
    // for (size_t bucket = 0; bucket < cpu_offsets.size(); ++bucket) {
    //     auto begin = cpu_offsets[bucket];
    //     auto end =
    //         (bucket + 1 == cpu_offsets.size()) ? size : cpu_offsets[bucket +
    //         1];
    //     __odd_even_sort<Float, NBlocks, NThreads>(grouped_data, begin, end);
    // }
    // for (size_t bucket = 0, nbuckets = cpu_offsets.size();
    //      bucket != nbuckets;) {
    //     while (bucket + 1 < nbuckets &&
    //            cpu_offsets[bucket + 1] - cpu_offsets[bucket] < 2)
    //         ++bucket;
    //     size_t bucket_end = bucket + 1;
    //     while (bucket_end + 1 < nbuckets &&
    //            cpu_offsets[bucket_end + 1] - cpu_offsets[bucket] <=
    //                2 * NThreads)
    //         ++bucket_end;
    //     auto begin = cpu_offsets[bucket];
    //     auto end = (bucket_end == nbuckets) ? size : cpu_offsets[bucket_end];
    //     __odd_even_sort<Float, NBlocks, NThreads>(grouped_data, begin, end);
    //     bucket = bucket_end;
    // }

    grouped_data.Populate(data);
}
}  // namespace sort

#endif