#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <csignal>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "equation.cuh"
#include "errors.cuh"
#include "signals.cuh"

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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // продвинутая магия cuda
    int ndevices;
    CHECK_CALL_ERRORS(cudaGetDeviceCount(&ndevices));
    int device = rank % ndevices;
    CHECK_CALL_ERRORS(cudaSetDevice(device));

    std::string output_filepath;
    equation::TaskParams tparams;
    equation::CalculationParams cparams;
    if (rank == 0) {
        equation::ReadTaskParams(std::cin, tparams);
        std::cin >> output_filepath;
        equation::ReadCalculationParams(std::cin, cparams);
#ifdef DEBUG
        debug_print_params(output_filepath, tparams, cparams);
#endif
    }
    equation::BcastOutputFilepath(output_filepath, MPI_COMM_WORLD);
    equation::BcastTaskParams(tparams, MPI_COMM_WORLD);
    equation::BcastCalculationParams(cparams, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto u = equation::SolveEquation(tparams, cparams, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    equation::WriteCalculationResults(output_filepath, u, tparams,
                                      MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}