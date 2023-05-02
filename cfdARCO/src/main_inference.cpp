// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <iostream>
#include <matplot/matplot.h>
#include <chrono>
#include <thread>

#include "mesh2d.hpp"
#include "fvm.hpp"
#include "cfdarcho_main.hpp"
#include "io_operators.hpp"


int main(int argc, char **argv) {
    CFDArcoGlobalInit::initialize(argc, argv, false);
    auto mesh = read_mesh();
    std::cout << "Mesh read" << std::endl;
    auto [vars, history_count] = init_read_history_stepping(mesh.get());
    std::cout << "Vars read" << std::endl;

    auto& rho = vars.at(0);

    auto fig = matplot::figure(true);
    for (int i = 0; i < history_count - 1; ++i) {
        if (i % 50 != 0) continue;
        std::cout << "Reading step " << i << std::endl;
        read_history_stepping(mesh.get(), {&rho}, i);
        std::cout << "Reading step " << i << " done" << std::endl;
        auto grid_hist = to_grid_local(mesh.get(), rho.current);
        auto vect = from_eigen_matrix<double>(grid_hist);
        fig->current_axes()->image(vect);
        fig->draw();
        std::this_thread::sleep_for(std::chrono::milliseconds {10});
    }

    CFDArcoGlobalInit::finalize();
    return 0;
}