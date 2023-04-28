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
    CFDArcoGlobalInit::initialize(argc, argv);
    auto [mesh, vars] = read_history();

    auto& rho = vars.at(0);

    auto fig = matplot::figure(true);
    for (int i = 0; i < rho.history.size() - 1; ++i) {
        if (i % 10 != 0) continue;
        auto grid_hist = to_grid_local(mesh.get(), rho.history[i]);
        auto vect = from_eigen_matrix<double>(grid_hist);
        fig->current_axes()->image(vect);
        fig->draw();
        std::this_thread::sleep_for(std::chrono::milliseconds {100});
    }

    CFDArcoGlobalInit::finalize();
    return 0;
}