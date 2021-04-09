#include <thrust/device_ptr.h>

#include <csignal>
#include <iostream>
#include <vector>

#ifndef NBLOCKS
#define NBLOCKS 256
#endif
#ifndef NTHREADS
#define NTHREADS 256
#endif

#define CSC(call)                                                      \
    do {                                                               \
        cudaError_t res = call;                                        \
        if (res != cudaSuccess) {                                      \
            fprintf(stderr, "ERROR in %s:%d. Message: %s\n", __FILE__, \
                    __LINE__, cudaGetErrorString(res));                \
            exit(0);                                                   \
        }                                                              \
    } while (0)

void signal_handler(int sig) { exit(0); }

void handle_signals() {
    std::signal(SIGSEGV, signal_handler);
    std::signal(SIGABRT, signal_handler);
}

using DataType = uint8_t;

std::vector<DataType> read_binary_array(std::istream& is) {
    uint32_t size;
    is.read(static_cast<char*>(static_cast<void*>(&size)), sizeof(size));
    std::vector<DataType> data(size);
    is.read(static_cast<char*>(static_cast<void*>(data.data())),
            size * sizeof(DataType));
    return data;
}

void write_binary_array(std::ostream& os, const std::vector<DataType>& data) {
    os.write(static_cast<const char*>(static_cast<const void*>(data.data())),
             data.size() * sizeof(DataType));
}

template <typename Type>
__global__ void device_memset(Type* data, uint32_t size, Type value) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t offset = blockDim.x * gridDim.x;
    for (uint32_t it = idx; it < size; it += offset) data[it] = value;
}

const uint32_t NBins = 1u << 8;

__global__ void histogram(uint32_t* histogram, const DataType* data,
                          uint32_t size) {
    __shared__ uint32_t local_histogram[NBins];
    for (uint32_t it = threadIdx.x; it < NBins; it += blockDim.x)
        local_histogram[it] = 0;
    __syncthreads();
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t offset = blockDim.x * gridDim.x;
    for (uint32_t it = idx; it < size; it += offset) {
        atomicAdd(&local_histogram[data[it]], uint32_t{1});
    }
    __syncthreads();
    for (uint32_t it = threadIdx.x; it < NBins; it += blockDim.x)
        atomicAdd(&histogram[it], local_histogram[it]);
}

const uint32_t NBanks = 32;

__device__ uint32_t bank_free_offset(uint32_t idx) { return idx / NBanks; }

__global__ void scan(uint32_t* data) {
    uint32_t thread = threadIdx.x;
    __shared__ uint32_t shared[NBins + NBins / NBanks];
    uint32_t ai = thread, bi = thread + NBins / 2;
    uint32_t offset_a = bank_free_offset(ai), offset_b = bank_free_offset(bi);
    shared[ai + offset_a] = data[ai];
    shared[bi + offset_b] = data[bi];
    for (uint32_t offset = 1; offset < NBins; offset <<= 1) {
        __syncthreads();
        uint32_t it = 2 * offset * thread, r_it = NBins - 1 - it;
        if (it + offset < NBins)
            shared[r_it + +bank_free_offset(r_it)] +=
                shared[r_it - offset + +bank_free_offset(r_it - offset)];
    }
    __syncthreads();
    if (thread == 0) shared[NBins - 1 + bank_free_offset(NBins - 1)] = 0;
    for (uint32_t offset = NBins / 2; offset != 0; offset >>= 1) {
        __syncthreads();
        uint32_t it = 2 * offset * thread, r_it = NBins - 1 - it;
        if (it + offset < NBins) {
            int32_t tmp = shared[r_it + bank_free_offset(r_it)];
            shared[r_it + bank_free_offset(r_it)] +=
                shared[r_it - offset + bank_free_offset(r_it - offset)];
            shared[r_it - offset + bank_free_offset(r_it - offset)] = tmp;
        }
    }
    __syncthreads();
    data[ai] = shared[ai + offset_a];
    data[bi] = shared[bi + offset_b];
}

__global__ void counting_sort_place(DataType* dst, const DataType* src,
                                    uint32_t size, uint32_t* offsets) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t offset = blockDim.x * gridDim.x;
    for (uint32_t it = idx; it < size; it += offset) {
        DataType value = src[it];
        uint32_t dst_idx = atomicAdd(&offsets[value], uint32_t{1});
        dst[dst_idx] = value;
    }
}

template <class Type>
struct Data {
    uint32_t size;
    Type* data;

    Data(uint32_t size, Type value) : size(size), data(nullptr) {
        CSC(cudaMalloc(&this->data, sizeof(Type) * this->size));
        device_memset<<<NBLOCKS, NTHREADS>>>(this->data, size, value);
        cudaDeviceSynchronize();
        CSC(cudaGetLastError());
    }

    Data(const std::vector<Type>& src) : size(src.size()), data(nullptr) {
        CSC(cudaMalloc(&this->data, sizeof(Type) * this->size));
        CSC(cudaMemcpy(this->data, src.data(), sizeof(Type) * this->size,
                       cudaMemcpyHostToDevice));
    }

    void ToHost(std::vector<Type>& dst) const {
        dst.resize(this->size);
        if (this->data != nullptr) {
            CSC(cudaMemcpy(dst.data(), this->data, sizeof(Type) * this->size,
                           cudaMemcpyDeviceToHost));
        }
    }

    ~Data() {
        if (this->data != nullptr) cudaFree(this->data);
    }
};

void counting_sort(std::vector<DataType>& data) {
    Data<DataType> gpu_data(data);
    Data<uint32_t> offsets(NBins, 0u);
    uint32_t size = gpu_data.size;

    histogram<<<NBLOCKS, NTHREADS>>>(offsets.data, gpu_data.data, size);
    cudaDeviceSynchronize();
    CSC(cudaGetLastError());

    scan<<<1, NBins / 2>>>(offsets.data);
    cudaDeviceSynchronize();
    CSC(cudaGetLastError());

    Data<DataType> gpu_sorted_data(size, 0u);
    counting_sort_place<<<NTHREADS, NBLOCKS>>>(
        gpu_sorted_data.data, gpu_data.data, size, offsets.data);
    cudaDeviceSynchronize();
    CSC(cudaGetLastError());

    gpu_sorted_data.ToHost(data);
}

int main() {
    handle_signals();
#ifndef DEBUG
    std::vector<DataType> data = read_binary_array(std::cin);
#else
    uint32_t n;
    std::cin >> n;
    std::vector<DataType> data(n);
    for (auto& it : data) {
        int value;
        std::cin >> value;
        it = value;
    }
#endif
    counting_sort(data);
#ifndef DEBUG
    write_binary_array(std::cout, data);
#else
    if (n < 1000) {
        for (auto it : data) {
            std::cerr << (int)it << ' ';
        }
        std::cerr << std::endl;
    }
#endif
}
