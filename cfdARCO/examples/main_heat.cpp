// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <iostream>
#include <matplot/matplot.h>
#include <chrono>
#include <thread>
#include <argparse/argparse.hpp>

#include "mesh2d.hpp"
#include "fvm.hpp"
#include "cfdarcho_main.hpp"
#include "io_operators.hpp"
#include "utils.hpp"
#include "val_utils.hpp"

Eigen::VectorXd initial_T(Mesh2D* mesh) {
    auto ret = Eigen::VectorXd{mesh->_num_nodes};
    ret.setConstant(0);
    sett(ret, mesh->_x, mesh->_y, mesh->_y * 0.25, mesh->_x * 0.25, 100.0);
    sett(ret, mesh->_x, mesh->_y, mesh->_y * 0.75, mesh->_x * 0.75, 100.0);
    return ret;
}

int main(int argc, char **argv) {
    SingleLibInitializer initializer{argc, argv};
    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    Eigen::VectorXd T_initial = initial_T(mesh.get());
    auto T = Variable(mesh.get(), T_initial, boundary_copy(T_initial), boundary_copy_cu(T_initial), "T");

    std::vector<Variable*> space_vars {&T};
    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu, 0.005, space_vars);

    std::vector<std::tuple<Variable*, char, Variable>> equation_system = {
            {d1t(T), '=', 5.0 * lapl(T)},
    };

    auto equation = Equation(timesteps);
    initializer.init_store(space_vars);

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(space_vars, equation_system, &dt, initializer.visualize, space_vars);
    auto end = std::chrono::steady_clock::now();
    if (CFDArcoGlobalInit::get_rank() == 0) std::cout << std::endl << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]" << std::endl;

    initializer.finalize();
    return 0;
}