#include <mpi.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

template <typename T>
struct Vector3 {
    T x, y, z;
};

using Vector3i = Vector3<int>;

struct NBlocks : Vector3i {};

struct BlockSize : Vector3i {};

struct TaskParams {
    NBlocks nblocks;
    BlockSize block_size;
};

struct CalculationParams {
    double eps;
    double lx, ly, lz;
    double down, up, left, right, front, back;
    double u0;
};

void read_task_params(std::istream& is, TaskParams& params) {
    is >> params.nblocks.x >> params.nblocks.y >> params.nblocks.z;
    is >> params.block_size.x >> params.block_size.y >> params.block_size.z;
}

void read_calculation_params(std::istream& is, CalculationParams& params) {
    is >> params.eps;
    is >> params.lx >> params.ly >> params.lz;
    is >> params.down >> params.up >> params.left >> params.right >>
        params.front >> params.back;
    is >> params.u0;
}

void bcast_task_params(TaskParams& params, MPI_Comm comm) {
    MPI_Bcast(&params.nblocks.x, 1, MPI_INT, 0, comm);
    MPI_Bcast(&params.nblocks.y, 1, MPI_INT, 0, comm);
    MPI_Bcast(&params.nblocks.z, 1, MPI_INT, 0, comm);
    MPI_Bcast(&params.block_size.x, 1, MPI_INT, 0, comm);
    MPI_Bcast(&params.block_size.y, 1, MPI_INT, 0, comm);
    MPI_Bcast(&params.block_size.z, 1, MPI_INT, 0, comm);
}

void bcast_calculation_params(CalculationParams& params, MPI_Comm comm) {
    MPI_Bcast(&params.eps, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&params.lx, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&params.ly, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&params.lz, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&params.down, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&params.up, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&params.left, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&params.right, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&params.front, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&params.back, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&params.u0, 1, MPI_DOUBLE, 0, comm);
}

#ifdef DEBUG
void debug_print_params(std::string output_filepath, const TaskParams& tparams,
                        const CalculationParams& cparams) {
    std::cerr << "output filepath: " << output_filepath << '\n';
    std::cerr << "task:\n";
    std::cerr << "\tnblocks: " << tparams.nblocks.x << ' ' << tparams.nblocks.y
              << ' ' << tparams.nblocks.z << '\n';
    std::cerr << "\tblocks sizes: " << tparams.block_size.x << ' '
              << tparams.block_size.y << ' ' << tparams.block_size.z << '\n';
    std::cerr << "calculation:\n";
    std::cerr << "\tl: " << cparams.lx << ' ' << cparams.ly << ' ' << cparams.lz
              << '\n';
    std::cerr << "\tu: " << cparams.down << ' ' << cparams.up << ' '
              << cparams.left << ' ' << cparams.right << ' ' << cparams.front
              << ' ' << cparams.back << '\n';
    std::cerr << "\tu0: " << cparams.u0 << std::endl;
}
#endif

int block_id(NBlocks nblocks, int x, int y, int z) {
    return ((z) * (nblocks.x * nblocks.y)) + y * nblocks.x + x;
}

int cell_id(BlockSize block_size, int x, int y, int z) {
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

double process_iteration(std::vector<double>& u_next, std::vector<double>& u,
                         const TaskParams& tparams,
                         const CalculationParams& cparams) {
    double hx = cparams.lx / (tparams.nblocks.x * tparams.block_size.x),
           hy = cparams.ly / (tparams.nblocks.y * tparams.block_size.y),
           hz = cparams.lz / (tparams.nblocks.z * tparams.block_size.z);
    double max_error = 0.0;
    for (int i = 0; i < tparams.block_size.x; ++i)
        for (int j = 0; j < tparams.block_size.y; ++j)
            for (int k = 0; k < tparams.block_size.z; ++k) {
                double inv_hxsqr = 1.0 / (hx * hx);
                double inv_hysqr = 1.0 / (hy * hy);
                double inv_hzsqr = 1.0 / (hz * hz);

                double num = (u[cell_id(tparams.block_size, i + 1, j, k)] +
                              u[cell_id(tparams.block_size, i - 1, j, k)]) *
                                 inv_hxsqr +
                             (u[cell_id(tparams.block_size, i, j + 1, k)] +
                              u[cell_id(tparams.block_size, i, j - 1, k)]) *
                                 inv_hysqr +
                             (u[cell_id(tparams.block_size, i, j, k + 1)] +
                              u[cell_id(tparams.block_size, i, j, k - 1)]) *
                                 inv_hzsqr;
                double denum = 2.0 * (inv_hxsqr + inv_hysqr + inv_hzsqr);
                double temp = num / denum;
                double error =
                    fabs(u[cell_id(tparams.block_size, i, j, k)] - temp);
                if (error > max_error) max_error = error;
                u_next[cell_id(tparams.block_size, i, j, k)] = temp;
            }
    return max_error;
}

enum Border : int { Left, Right, Back, Front, Down, Up };

struct BlockIndex : Vector3i {
    BlockIndex(int rank, const NBlocks& nblocks)
        : Vector3i{rank % (nblocks.x * nblocks.y) % nblocks.x,
                   rank % (nblocks.x * nblocks.y) / nblocks.x,
                   rank / nblocks.x / nblocks.y} {}
};

void exchange_border_conditions(std::vector<double>& u,
                                std::vector<double>& sendbuf,
                                std::vector<double>& recvbuf,
                                const TaskParams& tparams,
                                const CalculationParams& cparams,
                                MPI_Comm comm) {
    BlockIndex idx(comm_rank(comm), tparams.nblocks);
    MPI_Status status;
    if (idx.x > 0) {
        for (int j = 0; j < tparams.block_size.y; ++j)
            for (int k = 0; k < tparams.block_size.z; ++k)
                sendbuf[j * tparams.block_size.z + k] =
                    u[cell_id(tparams.block_size, 0, j, k)];

        int count = tparams.block_size.y * tparams.block_size.z;
        int coupled_process_rank =
            block_id(tparams.nblocks, idx.x - 1, idx.y, idx.z);

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

std::vector<double> solve_equation(const TaskParams& tparams,
                                   const CalculationParams& cparams,
                                   MPI_Comm comm) {
    int ncells = (tparams.block_size.x + 2) * (tparams.block_size.y + 2) *
                 (tparams.block_size.z + 2);
    std::vector<double> u(ncells, cparams.u0), u_next(ncells);
    int dim = std::max(tparams.block_size.x,
                       std::max(tparams.block_size.y, tparams.block_size.z));
    std::vector<double> sendbuf(dim * dim), recvbuf(dim * dim);
    double error = 1. / 0.;
    while (error > cparams.eps) {
        exchange_border_conditions(u, sendbuf, recvbuf, tparams, cparams, comm);
        MPI_Barrier(comm);
        double node_error = process_iteration(u_next, u, tparams, cparams);
        error = gather_error(node_error, comm);
        std::swap(u, u_next);
    }
    return u;
}

void write_calculation_results(std::string output_filepath,
                               const std::vector<double>& u,
                               const TaskParams& tparams, MPI_Comm comm) {
    int dim = std::max(tparams.block_size.x,
                       std::max(tparams.block_size.y, tparams.block_size.z));
    std::vector<double> buffer(dim * dim);
    int count = tparams.block_size.x;
    if (comm_rank(comm) == 0) {
        std::ofstream fout(output_filepath);
        MPI_Status status;

        for (int bk = 0; bk < tparams.nblocks.z; ++bk) {
            for (int k = 0; k < tparams.block_size.z; ++k) {
                for (int bj = 0; bj < tparams.nblocks.y; ++bj) {
                    for (int j = 0; j < tparams.block_size.y; ++j) {
                        for (int bi = 0; bi < tparams.nblocks.x; ++bi) {
                            int id = block_id(tparams.nblocks, bi, bj, bk);
                            if (id == 0) {
                                for (int i = 0; i < tparams.block_size.x; ++i)
                                    buffer[i] =
                                        u[cell_id(tparams.block_size, i, j, k)];
                            } else {
                                MPI_Recv(buffer.data(), count, MPI_DOUBLE, id,
                                         k * tparams.block_size.z + j, comm,
                                         &status);
                            }

                            for (int i = 0; i < tparams.block_size.x; ++i) {
                                fout << std::scientific << buffer[i] << ' ';
                            }
                        }
                        fout << '\n';
                    }
                }
                fout << '\n';
            }
        }
    } else {
        for (int k = 0; k < tparams.block_size.z; ++k) {
            for (int j = 0; j < tparams.block_size.y; ++j) {
                for (int i = 0; i < tparams.block_size.x; ++i)
                    buffer[i] = u[cell_id(tparams.block_size, i, j, k)];
                MPI_Send(buffer.data(), count, MPI_DOUBLE, 0,
                         k * tparams.block_size.z + j, comm);
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    std::string output_filepath;
    TaskParams tparams;
    CalculationParams cparams;
    if (comm_rank(MPI_COMM_WORLD) == 0) {
        read_task_params(std::cin, tparams);
        std::cin >> output_filepath;
        read_calculation_params(std::cin, cparams);
#ifdef DEBUG
        debug_print_params(output_filepath, tparams, cparams);
#endif
    }
    bcast_task_params(tparams, MPI_COMM_WORLD);
    bcast_calculation_params(cparams, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto u = solve_equation(tparams, cparams, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    write_calculation_results(output_filepath, u, tparams, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}