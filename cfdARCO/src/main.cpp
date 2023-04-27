// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <iostream>
#include <matplot/matplot.h>
#include <chrono>
#include <thread>

#include "mesh2d.hpp"
#include "fvm.hpp"
#include "cfdarcho_main.hpp"


Eigen::VectorXd initial_val(Mesh2D* mesh, double val_out, double val_in) {
    auto ret = Eigen::VectorXd{mesh->_num_nodes};
    int i = 0;
    for (auto& node : mesh->_nodes) {
        if (.3 < node->y() && node->y() < .7) {
            ret(i) = val_in;
        } else {
            ret(i) = val_out;
        }
        ++i;
    }
    return ret;
}

Eigen::VectorXd initial_pertrubations(Mesh2D* mesh, double val_out, double val_in) {
    auto ret = Eigen::VectorXd{mesh->_num_nodes};
    int i = 0;
    for (auto& node : mesh->_nodes) {
        if (.3 < node->x() && node->x() < .7) {
            ret(i) = val_in;
        } else {
            ret(i) = val_out;
        }
        ++i;
    }
    return ret;
}

Eigen::VectorXd boundary_none(Mesh2D* mesh, Eigen::VectorXd& arr) {
    return arr;
}

Eigen::VectorXd boundary_copy(Mesh2D* mesh, Eigen::VectorXd& arr, Eigen::VectorXd& copy_var) {
    auto arr1 = arr.cwiseProduct(mesh->_node_is_boundary_reverce);
    auto copy_var1 = copy_var.cwiseProduct(mesh->_node_is_boundary);
    return arr1 + copy_var1;
}

int main(int argc, char **argv) {
    CFDArcoGlobalInit::initialize(argc, argv);

    bool visualize = 1;

    size_t L = 300;
    size_t timesteps = 1000;
    double CFL = 0.5;
    double gamma = 5. / 3.;

    auto mesh = Mesh2D{L, L, 1, 1};
    mesh.init_basic_internals();
    mesh.compute();

    CFDArcoGlobalInit::make_node_distribution(&mesh, {10, 1, 1, 1, 1, 1, 1, 1, 1});

    if (CFDArcoGlobalInit::get_rank() < 2 ) {
        CFDArcoGlobalInit::enable_cuda(&mesh);
    }

    auto rho_initial = initial_val(&mesh, 1, 2);
    auto rho = Variable(&mesh,
                          rho_initial,
                          [& rho_initial] (Mesh2D* mesh, Eigen::VectorXd& arr) { return boundary_copy(mesh, arr, rho_initial); },
                          "rho");

    auto u_initial = initial_val(&mesh, -0.5, 0.5);
    auto u = Variable(&mesh,
                          u_initial,
                          [& u_initial] (Mesh2D* mesh, Eigen::VectorXd& arr) { return boundary_copy(mesh, arr, u_initial); },
                          "u");

    auto v_initial = initial_pertrubations(&mesh, -0.5, -0.3);
    auto v = Variable(&mesh,
                        v_initial,
                        [& v_initial] (Mesh2D* mesh, Eigen::VectorXd& arr) { return boundary_copy(mesh, arr, v_initial); },
                        "v");

    Eigen::VectorXd p_initial = Eigen::VectorXd{mesh._num_nodes};
    p_initial.setConstant(2.5);
    auto p = Variable(&mesh,
                        p_initial,
                        [& p_initial] (Mesh2D* mesh, Eigen::VectorXd& arr) { return boundary_copy(mesh, arr, p_initial); },
                        "p");

    Eigen::VectorXd mass_initial = rho.current.array() * mesh._volumes.array();
    auto mass = Variable(&mesh,
                           mass_initial,
                           [& mass_initial] (Mesh2D* mesh, Eigen::VectorXd& arr) { return boundary_copy(mesh, arr, mass_initial); },
                           "mass");

    Eigen::VectorXd rho_u_initial = rho.current.array() * u.current.array() * mesh._volumes.array();
    auto rho_u = Variable(&mesh,
                            rho_u_initial,
                           [& rho_u_initial] (Mesh2D* mesh, Eigen::VectorXd& arr) { return boundary_copy(mesh, arr, rho_u_initial); },
                           "rho_u");

    Eigen::VectorXd rho_v_initial = rho.current.array() * v.current.array() * mesh._volumes.array();
    auto rho_v = Variable(&mesh,
                            rho_v_initial,
                            [& rho_v_initial] (Mesh2D* mesh, Eigen::VectorXd& arr) { return boundary_copy(mesh, arr, rho_v_initial); },
                            "rho_v");

    auto E = p / (gamma - 1) + 0.5 * rho * ((u * u) + (v * v));
    Eigen::VectorXd E_initial = (p.current.array() / (gamma - 1) + 0.5 * rho.current.array() * (u.current.array() * u.current.array() + v.current.array() * v.current.array())) * mesh._volumes.array();
    auto rho_e = Variable(&mesh,
                          E_initial,
                          [& E_initial] (Mesh2D* mesh, Eigen::VectorXd& arr) { return boundary_copy(mesh, arr, E_initial); },
                          "rho_e");

    std::vector<Variable*> space_vars {&u, &v, &p, &rho};
    auto dt = DT(&mesh, UpdatePolicies::CourantFriedrichsLewy, CFL, space_vars);

    std::vector<std::tuple<Variable*, char, Variable>> equation_system = {
            {&rho,        '=', mass / mesh._volumes},
            {&u,          '=', rho_u / rho / mesh._volumes},
            {&v,          '=', rho_v / rho / mesh._volumes},
            {&p,          '=', (rho_e / mesh._volumes - 0.5 * rho * (u * u + v * v)) * (gamma - 1)},

            {&rho,    '=', rho - 0.5 * dt * (u * rho.dx() + rho * u.dx() + v * rho.dy() + rho * v.dy())},
            {&u,      '=', u - 0.5 * dt * (u * u.dx() + v * u.dy() + (1 / rho) * p.dx())},
            {&v,      '=', v - 0.5 * dt * (u * v.dx() + v * v.dy() + (1 / rho) * p.dy())},
            {&p,      '=', p - 0.5 * dt * (gamma * p * (u.dx() + v.dy()) + u * p.dx() + v * p.dy())},

            {d1t(mass),  '=', -((d1dx(rho * u) + d1dy(rho * v)) - stab_tot(rho) * 2)},
            {d1t(rho_u), '=', -((d1dx(rho * u * u + p) + d1dy(rho * v * u)) - stab_tot(rho * u) * 2)},
            {d1t(rho_v), '=', -((d1dx(rho * v * u) + d1dy(rho * v * v + p)) - stab_tot(rho * v) * 2)},
            {d1t(rho_e), '=', -((d1dx((E + p) * u) + d1dy((E + p) * v)) - stab_tot(E) * 2)},

    };

    auto equation = Equation(timesteps);

    std::vector<Variable*> all_vars {&rho, &u, &v, &p, &mass, &rho_u, &rho_v, &rho_e};

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, visualize);
    auto end = std::chrono::steady_clock::now();
    if (CFDArcoGlobalInit::get_rank() == 0) std::cout << std::endl << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]" << std::endl;

    if (visualize) {
        auto fig = matplot::figure(true);
        for (int i = 0; i < rho.history.size() - 1; ++i) {
            if (i % 10 != 0) continue;
            auto grid_hist = to_grid(&mesh, rho.history[i]);
            if (CFDArcoGlobalInit::get_rank() == 0) {
                auto vect = from_eigen_matrix<double>(grid_hist);
                fig->current_axes()->image(vect);
                fig->draw();
                std::this_thread::sleep_for(std::chrono::milliseconds {100});
            }
        }
    }

    CFDArcoGlobalInit::finalize();
    return 0;
}