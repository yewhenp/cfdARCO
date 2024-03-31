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

Eigen::VectorXd initial_h(Mesh2D* mesh, double H, double rossby_radius) {
    auto ret = Eigen::VectorXd{mesh->_num_nodes};
    int i = 0;
    for (auto& node : mesh->_nodes) {
        ret(i) = H + 1.0 * std::exp(
                - std::pow(node->x() - mesh->_dx * mesh->_lx / 2, 2) / std::pow(rossby_radius, 2)
                - std::pow(node->y() - mesh->_dy * mesh->_ly / 2, 2) / std::pow(rossby_radius, 2)
                );
        ++i;
    }
    return ret;
}

int main(int argc, char **argv) {
    SingleLibInitializer initializer{argc, argv};
    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;


    double g = 9.81;
    double f = 4e-4;
    double H = 100;
    double k = 0;
    double rossby_radius = std::sqrt(g * H) / f;

    Eigen::VectorXd u_initial = Eigen::VectorXd{mesh->_num_nodes};
    u_initial.setConstant(0);
    auto u = Variable(mesh.get(), u_initial, boundary_copy(u_initial), boundary_copy_cu(u_initial), "u_shallow");

    Eigen::VectorXd v_initial = Eigen::VectorXd{mesh->_num_nodes};
    v_initial.setConstant(0);
    auto v = Variable(mesh.get(), v_initial, boundary_copy(v_initial), boundary_copy_cu(v_initial), "v_shallow");

    Eigen::VectorXd h_initial = initial_h(mesh.get(), H, rossby_radius);
    auto h = Variable(mesh.get(), h_initial, boundary_none, boundary_none_cu, "h_shallow");

    std::vector<Variable*> space_vars {&u, &v, &h};
    double stable_dt = 0.5 * std::min(mesh->_lx * mesh->_dx, mesh->_ly * mesh->_dy) / std::sqrt(g * H);
    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu, 2, space_vars);



    std::vector<std::tuple<Variable*, char, Variable>> equation_system = {
            {d1t(h), '=', - H * (d1dx(u) + d1dy(v))},
            {d1t(u), '=', -g * d1dx(h) - k * u + f * v },
            {d1t(v), '=', -g * d1dy(h) - k * v - f * u  },
    };

    auto equation = Equation(timesteps);

    std::vector<Variable*> all_vars {&u, &v, &h};

    initializer.init_store({&h});

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, initializer.visualize, {&h});
    auto end = std::chrono::steady_clock::now();
    if (CFDArcoGlobalInit::get_rank() == 0) std::cout << std::endl << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]" << std::endl;

    initializer.finalize();
    return 0;
}