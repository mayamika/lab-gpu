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
    Float data_min_element_, data_max_element_;

public:
    __BucketIndexer(uint32_t nbuckets, Float min_element, Float max_element)
        : nbuckets_(nbuckets),
          data_min_element_(min_element),
          data_max_element_(max_element){};

    __host__ __device__ uint32_t operator()(Float value_to_index) const {
        Float bucket_width =
            (this->data_max_element_ - this->data_min_element_) /
            this->nbuckets_;
        uint32_t idx = static_cast<uint32_t>(
            (value_to_index - this->data_min_element_) / bucket_width);
        return __min(this->nbuckets_ - 1, idx);
    }

    __host__ __device__ uint32_t nbuckets() const { return this->nbuckets_; }
};

template <typename Float>
struct __DataLimits {
    Float min_element, max_element;
};

template <typename Float, size_t NBlocks, size_t NThreads>
__host__ __DataLimits<Float> __find_data_limits(const Float* data,
                                                size_t size) {
    thrust::device_ptr<const Float> data_device_ptr =
        thrust::device_pointer_cast(data);
    auto min_max = thrust::minmax_element(thrust::device, data_device_ptr,
                                          data_device_ptr + size);
    return {*min_max.first, *min_max.second};
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
    const Float* data, size_t size,
    const __BucketIndexer<Float>& bucket_indexer) {
    gpu::Vector<uint32_t> histogram =
        gpu::MakeVector<uint32_t, NBlocks, NThreads>(bucket_indexer.nbuckets(),
                                                     uint32_t{});
    __bucket_histogram_kernel<Float>
        <<<NBlocks, NThreads>>>(histogram.Data(), data, size, bucket_indexer);
    CHECK_KERNEL_ERRORS();
    return histogram;
}

template <typename T, size_t NThreads>
__global__ void __partial_blelloch_scan(T* data, int64_t size,
                                        T* partial_sums) {
    int64_t tidx = threadIdx.x;
    for (int64_t segment_begin = blockIdx.x * 2 * blockDim.x;
         segment_begin < size; segment_begin += 2 * blockDim.x * gridDim.x) {
        int64_t segment_end = __min(segment_begin + 2 * blockDim.x, size);
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
            if (partial_sums != nullptr)
                partial_sums[segment_begin / (2 * blockDim.x)] =
                    shared[segment_size - 1];
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
        data[i] += partial_sums[i / (2 * blockDim.x)];
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
    if (size <= 2 * NThreads) {
        // Запустим ту же реализацию скана.
        // Для size <= 2 * NThreads мы получим обычный скан.
        __partial_blelloch_scan<T, NThreads>
            <<<1, NThreads>>>(data.Data(), data.Size(), nullptr);
        CHECK_KERNEL_ERRORS();
        return;
    }
    // Рекурсивная часть алгоритма
    // Посчитаем скан от частичных сумм, и отсканим суммы с помощью этого же
    // алгортима.
    gpu::Vector<T> partial_sums = gpu::MakeVector<T, NBlocks, NThreads>(
        (size + 2 * NThreads - 1) / (2 * NThreads), T{});
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
    const Float* data, size_t size,
    const __BucketIndexer<Float>& bucket_indexer) {
    gpu::Vector<uint32_t> bucket_histogram =
        __bucket_histogram<Float, NBlocks, NThreads>(data, size,
                                                     bucket_indexer);
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
__host__ void __bucket_group(Float* data, size_t size,
                             const gpu::Vector<uint32_t>& offsets,
                             const __BucketIndexer<Float>& bucket_indexer) {
    // Дополнительная гистограмма, которая будет хранить размеры бакетов.
    gpu::Vector<uint32_t> sizes = __bucket_histogram<Float, NBlocks, NThreads>(
        data, size, bucket_indexer);
    gpu::Vector<Float> dst_data =
        gpu::MakeVector<Float, NBlocks, NThreads>(size, Float{});
    cudaDeviceSynchronize();
    __bucket_group_kernel<Float>
        <<<NBlocks, NThreads>>>(dst_data.Data(), data, size, offsets.Data(),
                                sizes.Data(), bucket_indexer);
    cudaDeviceSynchronize();
    CHECK_KERNEL_ERRORS();
    CHECK_CALL_ERRORS(cudaMemcpy(data, dst_data.Data(), sizeof(Float) * size,
                                 cudaMemcpyDeviceToDevice););
}

template <typename T>
__global__ void __partial_odd_even_sort_kernel(T* data, const uint32_t* begins,
                                               const uint32_t* ends,
                                               int64_t nbuckets) {
    int64_t tidx = threadIdx.x;
    for (int64_t bucket = blockIdx.x; bucket < nbuckets; bucket += gridDim.x) {
        int64_t segment_begin = begins[bucket];
        int64_t segment_end = ends[bucket];
        int64_t segment_size = segment_end - segment_begin;
        for (size_t i = 0; i < segment_size; ++i) {
            __syncthreads();
            for (size_t j = segment_begin + 2 * tidx + (!(i & 1));
                 j + 1 < segment_end; j += 2 * blockDim.x) {
                if (data[j] > data[j + 1]) {
                    T t = data[j];
                    data[j] = data[j + 1];
                    data[j + 1] = t;
                }
            }
        }
    }
}

template <typename T, size_t NBlocks, size_t NThreads>
__host__ void __partial_odd_even_sort(T* data, const uint32_t* begins,
                                      const uint32_t* ends, size_t nbuckets) {
    __partial_odd_even_sort_kernel<T>
        <<<NBlocks, NThreads>>>(data, begins, ends, nbuckets);
    CHECK_KERNEL_ERRORS();
}

template <typename Float, size_t NBlocks, size_t NThreads>
__host__ void __recursive_bucket_sort(Float* data, size_t size) {
    if (size < 2) {
        return;
    }
    __DataLimits<Float> data_limits =
        __find_data_limits<Float, NBlocks, NThreads>(data, size);
    if (data_limits.min_element == data_limits.max_element) {
        return;
    }
    // Выбирем количество сплитов таким образом,
    // чтобы в каждый сплит попало не более 2 * NThreads элементов.
    // Позже мы склеим сплиты в бакеты, так что можно взять число побольше.
    uint32_t nsplits = size;
    __BucketIndexer<Float> bucket_indexer(nsplits, data_limits.min_element,
                                          data_limits.max_element);
    gpu::Vector<uint32_t> offsets =
        __bucket_offsets<Float, NBlocks, NThreads>(data, size, bucket_indexer);
    cudaDeviceSynchronize();
    // Сгруппируем элементы массива согласно разбиению.
    __bucket_group<Float, NBlocks, NThreads>(data, size, offsets,
                                             bucket_indexer);
    cudaDeviceSynchronize();
    // Теперь склеим маленькие бакеты на cpu для эффективной
    // сортировки чет-нечет, за одно убедимся, что все бакеты можно обработать
    // одноблочной сортировкой.
    std::vector<uint32_t> cpu_offsets = offsets.Host();
    offsets.Clear();
    std::vector<uint32_t> bucket_begins, bucket_ends;
    for (size_t split = 0; split != nsplits;) {
        auto split_end = split + 1;
        while (split_end != nsplits &&
               cpu_offsets[split_end] - cpu_offsets[split] <= 2 * NThreads)
            ++split_end;
        split_end = __max(split + 1, split_end - 1);
        auto segment_begin = cpu_offsets[split];
        auto segment_end =
            (split_end == nsplits) ? size : cpu_offsets[split_end];
        auto segment_size = segment_end - segment_begin;
        if (segment_size > 2 * NThreads) {
            __recursive_bucket_sort<Float, NBlocks, NThreads>(
                data + segment_begin, segment_size);
        } else {
            bucket_begins.push_back(segment_begin);
            bucket_ends.push_back(segment_end);
        }
        split = split_end;
    }
    size_t nbuckets = bucket_begins.size();
    gpu::Vector<uint32_t> gpu_bucket_begins(bucket_begins);
    gpu::Vector<uint32_t> gpu_bucket_ends(bucket_ends);
    __partial_odd_even_sort<Float, NBlocks, NThreads>(
        data, gpu_bucket_begins.Data(), gpu_bucket_ends.Data(), nbuckets);
}

template <typename Float, size_t NBlocks = 256, size_t NThreads = 256>
void BucketSort(std::vector<Float>& data) {
    static_assert(std::is_floating_point<Float>::value,
                  "only floating-point types allowed");
    gpu::Vector<Float> gpu_data(data);
    __recursive_bucket_sort<Float, NBlocks, NThreads>(gpu_data.Data(),
                                                      gpu_data.Size());
    gpu_data.Populate(data);
}
}  // namespace sort

#endif