#ifndef EQUATION_CUH
#define EQUATION_CUH

#include <mpi.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include <cmath>
#include <vector>

#include "gpu_vector.cuh"
#include "params.cuh"
#include "vector.cuh"

#ifndef NBLOCKS
#define NBLOCKS 256
#endif

#ifndef NTHREADS
#define NTHREADS 256
#endif

namespace equation {
int block_id(const NBlocks& nblocks, int x, int y, int z) {
    return ((z) * (nblocks.x * nblocks.y)) + y * nblocks.x + x;
}

int cell_id(const BlockSize& block_size, int x, int y, int z) {
    return (z + 1) * ((block_size.x + 2) * (block_size.y + 2)) +
           (y + 1) * (block_size.x + 2) + (x + 1);
}

int comm_rank(MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
}

int comm_size(MPI_Comm comm) {
    int size;
    MPI_Comm_size(comm, &size);
    return size;
}

double gather_error(double node_error, MPI_Comm comm) {
    int nprocesses = comm_size(comm);
    std::vector<double> errors(nprocesses);
    MPI_Allgather(&node_error, 1, MPI_DOUBLE, errors.data(), 1, MPI_DOUBLE,
                  comm);
    double error = node_error;
    for (auto it : errors) error = std::max(error, it);
    return error;
}

__global__ void copy_xy_plane_kernel(double* data, double* plane,
                                     BlockSize block_size, int k, bool toPlane,
                                     double boundary) {
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_y = blockIdx.y * blockDim.y + threadIdx.y;

    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    if (toPlane) {
        for (int i = id_x; i < block_size.x; i += offset_x)
            for (int j = id_y; j < block_size.y; j += offset_y)
                plane[i + j * block_size.x] =
                    data[cell_id(block_size, i, j, k)];
    } else {
        if (plane != nullptr) {
            for (int i = id_x; i < block_size.x; i += offset_x)
                for (int j = id_y; j < block_size.y; j += offset_y)
                    data[cell_id(block_size, i, j, k)] =
                        plane[i + j * block_size.x];
        } else {
            for (int i = id_x; i < block_size.x; i += offset_x)
                for (int j = id_y; j < block_size.y; j += offset_y)
                    data[cell_id(block_size, i, j, k)] = boundary;
        }
    }
}

__global__ void copy_xz_plane_kernel(double* data, double* plane,
                                     BlockSize block_size, int j, bool toPlane,
                                     double boundary) {
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_y = blockIdx.y * blockDim.y + threadIdx.y;

    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    if (toPlane) {
        for (int i = id_x; i < block_size.x; i += offset_x)
            for (k = id_y; k < block_size.z; k += offset_y)
                plane[i + k * block_size.x] =
                    data[cell_id(block_size, i, j, k)];
    } else {
        if (plane != nullptr) {
            for (int i = id_x; i < nx; i += offset_x)
                for (int k = id_y; k < nz; k += offset_y)
                    data[cell_id(block_size, i, j, k)] = plane[i + k * nx];
        } else {
            for (int i = id_x; i < block_size.x; i += offset_x)
                for (int k = id_y; k < block_size.z; k += offset_y)
                    data[cell_id(block_size, i, j, k)] = boundary;
        }
    }
}

__global__ void copy_yz_plane_kernel(double* data, double* plane,
                                     BlockSize block_size, int i, bool toPlane,
                                     double boundary) {
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_y = blockIdx.y * blockDim.y + threadIdx.y;

    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    if (toPlane) {
        for (int k = id_x; k < block_size.z; k += offset_x)
            for (int j = id_y; j < block_size.y; j += offset_y)
                plane[j + k * block_size.y] =
                    data[cell_id(block_size, i, j, k)];
    } else {
        if (plane != nullptr) {
            for (int k = id_x; k < block_size.z; k += offset_x)
                for (int j = id_y; j < block_size.y; j += offset_y)
                    data[cell_id(block_size, i, j, k)] =
                        plane[j + k * block_size.y];
        } else {
            for (int k = id_x; k < block_size.z; k += offset_x)
                for (int j = id_y; j < block_size.y; j += offset_y)
                    data[cell_id(block_size, i, j, k)] = boundary;
        }
    }
}

__global__ void iteration_kernel(double* u_next, double* u, TaskParams tparams,
                                 CalculationParams cparams) {
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    int id_z = threadIdx.z + blockIdx.z * blockDim.z;

    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;
    int offset_z = blockDim.z * gridDim.z;

    double hx = cparams.lx / (tparams.nblocks.x * tparams.block_size.x),
           hy = cparams.ly / (tparams.nblocks.y * tparams.block_size.y),
           hz = cparams.lz / (tparams.nblocks.z * tparams.block_size.z);

    for (int i = id_x; i < tparams.block_size.x; i += offset_x)
        for (int j = id_y; j < tparams.block_size.y; j += offset_y)
            for (int k = id_z; k < tparams.block_size.z; k += offset_z) {
                double inv_hxsqr = 1.0 / (hx * hx);
                double inv_hysqr = 1.0 / (hy * hy);
                double inv_hzsqr = 1.0 / (hz * hz);

                double val = (u[cell_id(tparams.block_size, i + 1, j, k)] +
                              u[cell_id(tparams.block_size, i - 1, j, k)]) *
                                 inv_hxsqr +
                             (u[cell_id(tparams.block_size, i, j + 1, k)] +
                              u[cell_id(tparams.block_size, i, j - 1, k)]) *
                                 inv_hysqr +
                             (u[cell_id(tparams.block_size, i, j, k + 1)] +
                              u[cell_id(tparams.block_size, i, j, k - 1)]) *
                                 inv_hzsqr;
                double denum = 2.0 * (inv_hxsqr + inv_hysqr + inv_hzsqr);
                u_next[cell_id(tparams.block_size, i, j, k)] = val / denum;
            }
}

__global__ void errors_kernel(double* errors, double* u_next, double* u,
                              BlockSize block_size) {
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    int id_z = threadIdx.z + blockIdx.z * blockDim.z;

    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;
    int offset_z = blockDim.z * gridDim.z;

    for (int i = id_x - 1; i < block_size.x + 1; i += offset_x)
        for (int j = id_y - 1; j < block_size.y + 1; j += offset_y)
            for (int k = id_z - 1; k < block_size.z + 1; k += offset_z)
                errors[cell_id(block_size, i, j, k)] =
                    fabs(u_next[cell_id(block_size, i, j, k)] -
                         u[cell_id(block_size, i, j, k)]);
}

enum Border : int { Left, Right, Back, Front, Down, Up };

struct BlockIndex : vector::Vector3i {
    BlockIndex(int rank, const NBlocks& nblocks)
        : vector::Vector3i{rank % (nblocks.x * nblocks.y) % nblocks.x,
                           rank % (nblocks.x * nblocks.y) / nblocks.x,
                           rank / nblocks.x / nblocks.y} {}
};

void exchange_border_conditions(gpu::Vector<double>& u,
                                gpu::Vector<double>& sendbuf,
                                gpu::Vector<double>& recvbuf,
                                const TaskParams& tparams,
                                const CalculationParams& cparams,
                                MPI_Comm comm) {
    BlockIndex idx(comm_rank(comm), tparams.nblocks);
    MPI_Status status;
    if (idx.x > 0) {
        copy_yz_plane_kernel(u.Data(), sendbuf.Data(), tparams.block_size, 0,
                             true, cparams.u0);
        copy_yz_plane_kernel(u.Data(), recvbuf.Data(),
                             tparams.block_size, ) int count =
            tparams.block_size.y * tparams.block_size.z;
        int coupled_process_rank =
            block_id(tparams.nblocks, idx.x - 1, idx.y, idx.z);

        CHECK_CALL_ERRORS(cudaDeviceSynchronize());
        CHECK_KERNEL_ERRORS();

        MPI_Request sendreq, recvreq;
        MPI_Isend(sendbuf.data(), count, MPI_DOUBLE, coupled_process_rank,
                  Border::Left, comm, &sendreq);
        MPI_Irecv(recvbuf.data(), count, MPI_DOUBLE, coupled_process_rank,
                  Border::Right, comm, &recvreq);
        MPI_Wait(&sendreq, &status);
        MPI_Wait(&recvreq, &status);

        for (int j = 0; j < tparams.block_size.y; ++j)
            for (int k = 0; k < tparams.block_size.z; ++k)
                u[cell_id(tparams.block_size, -1, j, k)] =
                    recvbuf[j * tparams.block_size.z + k];
    } else {
        for (int j = 0; j < tparams.block_size.y; ++j)
            for (int k = 0; k < tparams.block_size.z; ++k)
                u[cell_id(tparams.block_size, -1, j, k)] = cparams.left;
    }

    if (idx.x < tparams.nblocks.x - 1) {
        for (int j = 0; j < tparams.block_size.y; ++j)
            for (int k = 0; k < tparams.block_size.z; ++k)
                sendbuf[j * tparams.block_size.z + k] = u[cell_id(
                    tparams.block_size, tparams.block_size.x - 1, j, k)];

        int count = tparams.block_size.y * tparams.block_size.z;
        int coupled_process_rank =
            block_id(tparams.nblocks, idx.x + 1, idx.y, idx.z);

        MPI_Request sendreq, recvreq;
        MPI_Isend(sendbuf.data(), count, MPI_DOUBLE, coupled_process_rank,
                  Border::Right, comm, &sendreq);
        MPI_Irecv(recvbuf.data(), count, MPI_DOUBLE, coupled_process_rank,
                  Border::Left, comm, &recvreq);
        MPI_Wait(&sendreq, &status);
        MPI_Wait(&recvreq, &status);

        for (int j = 0; j < tparams.block_size.y; ++j)
            for (int k = 0; k < tparams.block_size.z; ++k)
                u[cell_id(tparams.block_size, tparams.block_size.x, j, k)] =
                    recvbuf[j * tparams.block_size.z + k];
    } else {
        for (int j = 0; j < tparams.block_size.y; ++j)
            for (int k = 0; k < tparams.block_size.z; ++k)
                u[cell_id(tparams.block_size, tparams.block_size.x, j, k)] =
                    cparams.right;
    }

    if (idx.y > 0) {
        for (int i = 0; i < tparams.block_size.x; ++i)
            for (int k = 0; k < tparams.block_size.z; ++k)
                sendbuf[i * tparams.block_size.z + k] =
                    u[cell_id(tparams.block_size, i, 0, k)];

        int count = tparams.block_size.x * tparams.block_size.z;
        int coupled_process_rank =
            block_id(tparams.nblocks, idx.x, idx.y - 1, idx.z);

        MPI_Request sendreq, recvreq;
        MPI_Isend(sendbuf.data(), count, MPI_DOUBLE, coupled_process_rank,
                  Border::Front, comm, &sendreq);
        MPI_Irecv(recvbuf.data(), count, MPI_DOUBLE, coupled_process_rank,
                  Border::Back, comm, &recvreq);
        MPI_Wait(&sendreq, &status);
        MPI_Wait(&recvreq, &status);

        for (int i = 0; i < tparams.block_size.x; ++i)
            for (int k = 0; k < tparams.block_size.z; ++k)
                u[cell_id(tparams.block_size, i, -1, k)] =
                    recvbuf[i * tparams.block_size.z + k];
    } else {
        for (int i = 0; i < tparams.block_size.x; ++i)
            for (int k = 0; k < tparams.block_size.z; ++k)
                u[cell_id(tparams.block_size, i, -1, k)] = cparams.front;
    }

    if (idx.y < tparams.nblocks.y - 1) {
        for (int i = 0; i < tparams.block_size.x; ++i)
            for (int k = 0; k < tparams.block_size.z; ++k)
                sendbuf[i * tparams.block_size.z + k] = u[cell_id(
                    tparams.block_size, i, tparams.block_size.y - 1, k)];

        int count = tparams.block_size.x * tparams.block_size.z;
        int coupled_process_rank =
            block_id(tparams.nblocks, idx.x, idx.y + 1, idx.z);

        MPI_Request sendreq, recvreq;
        MPI_Isend(sendbuf.data(), count, MPI_DOUBLE, coupled_process_rank,
                  Border::Back, comm, &sendreq);
        MPI_Irecv(recvbuf.data(), count, MPI_DOUBLE, coupled_process_rank,
                  Border::Front, comm, &recvreq);
        MPI_Wait(&sendreq, &status);
        MPI_Wait(&recvreq, &status);

        for (int i = 0; i < tparams.block_size.x; ++i)
            for (int k = 0; k < tparams.block_size.z; ++k)
                u[cell_id(tparams.block_size, i, tparams.block_size.y, k)] =
                    recvbuf[i * tparams.block_size.z + k];
    } else {
        for (int i = 0; i < tparams.block_size.x; ++i)
            for (int k = 0; k < tparams.block_size.z; ++k)
                u[cell_id(tparams.block_size, i, tparams.block_size.y, k)] =
                    cparams.back;
    }

    if (idx.z > 0) {
        for (int i = 0; i < tparams.block_size.x; ++i)
            for (int j = 0; j < tparams.block_size.y; ++j)
                sendbuf[i * tparams.block_size.y + j] =
                    u[cell_id(tparams.block_size, i, j, 0)];

        int count = tparams.block_size.x * tparams.block_size.y;
        int coupled_process_rank =
            block_id(tparams.nblocks, idx.x, idx.y, idx.z - 1);

        MPI_Request sendreq, recvreq;
        MPI_Isend(sendbuf.data(), count, MPI_DOUBLE, coupled_process_rank,
                  Border::Down, comm, &sendreq);
        MPI_Irecv(recvbuf.data(), count, MPI_DOUBLE, coupled_process_rank,
                  Border::Up, comm, &recvreq);
        MPI_Wait(&sendreq, &status);
        MPI_Wait(&recvreq, &status);

        for (int i = 0; i < tparams.block_size.x; ++i)
            for (int j = 0; j < tparams.block_size.y; ++j)
                u[cell_id(tparams.block_size, i, j, -1)] =
                    recvbuf[i * tparams.block_size.y + j];
    } else {
        for (int i = 0; i < tparams.block_size.x; ++i)
            for (int j = 0; j < tparams.block_size.y; ++j)
                u[cell_id(tparams.block_size, i, j, -1)] = cparams.down;
    }

    if (idx.z < tparams.nblocks.z - 1) {
        for (int i = 0; i < tparams.block_size.x; ++i)
            for (int j = 0; j < tparams.block_size.y; ++j)
                sendbuf[i * tparams.block_size.y + j] = u[cell_id(
                    tparams.block_size, i, j, tparams.block_size.z - 1)];

        int count = tparams.block_size.x * tparams.block_size.y;
        int coupled_process_rank =
            block_id(tparams.nblocks, idx.x, idx.y, idx.z + 1);

        MPI_Request sendreq, recvreq;
        MPI_Isend(sendbuf.data(), count, MPI_DOUBLE, coupled_process_rank,
                  Border::Up, comm, &sendreq);
        MPI_Irecv(recvbuf.data(), count, MPI_DOUBLE, coupled_process_rank,
                  Border::Down, comm, &recvreq);
        MPI_Wait(&sendreq, &status);
        MPI_Wait(&recvreq, &status);

        for (int i = 0; i < tparams.block_size.x; ++i)
            for (int j = 0; j < tparams.block_size.y; ++j)
                u[cell_id(tparams.block_size, i, j, tparams.block_size.z)] =
                    recvbuf[i * tparams.block_size.y + j];
    } else {
        for (int i = 0; i < tparams.block_size.x; ++i)
            for (int j = 0; j < tparams.block_size.y; ++j)
                u[cell_id(tparams.block_size, i, j, tparams.block_size.z)] =
                    cparams.up;
    }
}

std::vector<double> SolveEquation(const TaskParams& tparams,
                                  const CalculationParams& cparams,
                                  MPI_Comm comm) {
    int ncells = (tparams.block_size.x + 2) * (tparams.block_size.y + 2) *
                 (tparams.block_size.z + 2);
    gpu::Vector<double> u = gpu::MakeVector<double, NTHREADS, NBLOCKS>(
                            ncells, cparams.u0),
                        u_next =
                            gpu::MakeVector<double, NTHREADS, NBLOCKS>(ncells);
    int dim = std::max(tparams.block_size.x,
                       std::max(tparams.block_size.y, tparams.block_size.z));
    std::vector<double> sendbuf(dim * dim), recvbuf(dim * dim);
    double error = 1. / 0.;
    gpu::Vector<double> errors = gpu::MakeVector<double, NBLOCKS, NTHREADS>(
        (tparams.block_size.x + 2) * (tparams.block_size.y + 2) *
        (tparams.block_size.z + 2));
    while (error > cparams.eps) {
        exchange_border_conditions(u, sendbuf, recvbuf, tparams, cparams, comm);
        MPI_Barrier(comm);

        dim3 grid_dim((int)pow(NBLOCKS, 1 / 3), (int)pow(NBLOCKS, 1 / 3),
                      (int)pow(NBLOCKS, 1 / 3));
        dim3 block_dim((int)pow(NTHREADS, 1 / 3), (int)pow(NTHREADS, 1 / 3),
                       (int)pow(NTHREADS, 1 / 3));

        cudaDeviceSynchronize();
        iteration_kernel<<<grid_dim, block_dim>>>(u_next, u, tparams, cparams);
        cudaDeviceSynchronize();
        CHECK_KERNEL_ERRORS();
        errors_kernel<<<grid_dim, block_dim>>>(
            errors.Data(), gpu_u_next.Data(), gpu_u.Data(), tparams.block_size);
        cudaDeviceSynchronize();
        CHECK_KERNEL_ERRORS();
        thrust::device_ptr<double> errors_ptr =
            thrust::device_pointer_cast(errors.Data());
        auto node_error_ptr =
            thrust::max_element(errors_ptr, errors_ptr + errors.Size());

        error = gather_error(*node_error_ptr, comm);
        std::swap(u, u_next);
    }
    return u.Host();
}

void WriteCalculationResults(std::string output_filepath,
                             const std::vector<double>& u,
                             const TaskParams& tparams, MPI_Comm comm) {
    int buff_size = (tparams.block_size.x + 2) * (tparams.block_size.y + 2) *
                    (tparams.block_size.z + 2);
    int word_size = snprintf(NULL, 0, "% e ", 1.234567890);
    std::vector<char> buff(buff_size * word_size, ' ');
    for (int k = 0; k < tparams.block_size.z; ++k) {
        for (int j = 0; j < tparams.block_size.y; ++j) {
            int new_symbol_length;
            int i = 0;
            for (; i < tparams.block_size.x - 1; ++i) {
                new_symbol_length = sprintf(
                    &buff[cell_id(tparams.block_size, i, j, k) * word_size],
                    "% e", u[cell_id(tparams.block_size, i, j, k)]);
                if (new_symbol_length < word_size) {
                    buff[cell_id(tparams.block_size, i, j, k) * word_size +
                         new_symbol_length] = ' ';
                }
            }
            new_symbol_length =
                sprintf(&buff[cell_id(tparams.block_size, i, j, k) * word_size],
                        "% e\n", u[cell_id(tparams.block_size, i, j, k)]);
            if (new_symbol_length < word_size) {
                buff[cell_id(tparams.block_size, i, j, k) * word_size +
                     new_symbol_length] = ' ';
            }
        }
    }
    MPI_Datatype word_type, mem_type, file_type;
    MPI_Type_contiguous(word_size, MPI_CHAR, &word_type);
    MPI_Type_commit(&word_type);
    BlockIndex block_idx(comm_rank(comm), tparams.nblocks);
    int sizes[3] = {tparams.block_size.x + 2, tparams.block_size.y + 2,
                    tparams.block_size.z + 2},
        block_sizes[3] = {tparams.block_size.x, tparams.block_size.y,
                          tparams.block_size.z},
        starts[3] = {1, 1, 1},
        f_sizes[3] = {tparams.block_size.x * tparams.nblocks.x,
                      tparams.block_size.y * tparams.nblocks.y,
                      tparams.block_size.z * tparams.nblocks.z},
        f_starts[3] = {tparams.block_size.x * block_idx.x,
                       tparams.block_size.y * block_idx.y,
                       tparams.block_size.z * block_idx.z};
    MPI_Type_create_subarray(3, sizes, block_sizes, starts, MPI_ORDER_FORTRAN,
                             word_type, &mem_type);
    MPI_Type_commit(&mem_type);
    MPI_Type_create_subarray(3, f_sizes, block_sizes, f_starts,
                             MPI_ORDER_FORTRAN, word_type, &file_type);
    MPI_Type_commit(&file_type);
    MPI_File file;
    MPI_File_open(comm, output_filepath.data(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    MPI_File_set_view(file, 0, MPI_CHAR, file_type, "native", MPI_INFO_NULL);
    MPI_File_write_all(file, buff.data(), 1, mem_type, MPI_STATUS_IGNORE);
    MPI_Barrier(comm);
    MPI_File_close(&file);
    MPI_Barrier(comm);

    MPI_Type_free(&word_type);
    MPI_Type_free(&mem_type);
    MPI_Type_free(&file_type);
}
}  // namespace equation

#endif