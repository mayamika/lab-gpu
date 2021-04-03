#ifndef PARAMS_CUH
#define PARAMS_CUH

#include "vector.cuh"

namespace equation {
struct NBlocks : vector::Vector3i {};

struct BlockSize : vector::Vector3i {};

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

void ReadTaskParams(std::istream& is, TaskParams& params) {
    is >> params.nblocks.x >> params.nblocks.y >> params.nblocks.z;
    is >> params.block_size.x >> params.block_size.y >> params.block_size.z;
}

void ReadCalculationParams(std::istream& is, CalculationParams& params) {
    is >> params.eps;
    is >> params.lx >> params.ly >> params.lz;
    is >> params.down >> params.up >> params.left >> params.right >>
        params.front >> params.back;
    is >> params.u0;
}

void BcastOutputFilepath(std::string& output_filepath, MPI_Comm comm) {
    int size = output_filepath.size();
    MPI_Bcast(&size, 1, MPI_INT, 0, comm);
    output_filepath.resize(size);
    MPI_Bcast((char*)output_filepath.c_str(), size, MPI_CHAR, 0, comm);
}

void BcastTaskParams(TaskParams& params, MPI_Comm comm) {
    MPI_Bcast(&params.nblocks.x, 1, MPI_INT, 0, comm);
    MPI_Bcast(&params.nblocks.y, 1, MPI_INT, 0, comm);
    MPI_Bcast(&params.nblocks.z, 1, MPI_INT, 0, comm);
    MPI_Bcast(&params.block_size.x, 1, MPI_INT, 0, comm);
    MPI_Bcast(&params.block_size.y, 1, MPI_INT, 0, comm);
    MPI_Bcast(&params.block_size.z, 1, MPI_INT, 0, comm);
}

void BcastCalculationParams(CalculationParams& params, MPI_Comm comm) {
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
}  // namespace equation

#endif