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
#include "utils.hpp"
#include "val_utils.hpp"


inline void sett(Eigen::VectorXd& v, int rows, int cols, int x, int y, double val) {
    const size_t idx = x * rows + y;
    v[idx] = val;
}


Eigen::VectorXd boundary_sine(Mesh2D* mesh, Eigen::VectorXd& arr) {
    static int ii = 0;

    ++ii;

    Eigen::VectorXd ret{arr};
    sett(ret, mesh->_x, mesh->_y, mesh->_x * 0.25, mesh->_y * 0.25, std::sin(static_cast<double>(ii) * 0.01) * 1000);
    sett(ret, mesh->_x, mesh->_y, mesh->_x * 0.75, mesh->_y * 0.75, std::sin(static_cast<double>(ii) * 0.01) * 1000);

    return ret;
}

CudaDataMatrix boundary_sine_cu(Mesh2D* mesh, CudaDataMatrix& arr) {
    static int ii = 0;
    ++ii;

    CudaDataMatrix arr_n{arr};
    arr_n.set(mesh->_x, mesh->_y, mesh->_x * 0.25, mesh->_y * 0.25, std::sin(static_cast<double>(ii) * 0.01) * 1000);
    arr_n.set(mesh->_x, mesh->_y, mesh->_x * 0.75, mesh->_y * 0.75, std::sin(static_cast<double>(ii) * 0.01) * 1000);

    return arr_n;
}



int main(int argc, char **argv) {
    SingleLibInitializer initializer{argc, argv};
    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    auto initial_zero = initial_with_val(mesh.get(), 0);


    auto v_x = Variable(mesh.get(), initial_zero, boundary_sine, boundary_sine_cu, "v_x");
    auto v_y = Variable(mesh.get(), initial_zero, boundary_sine, boundary_sine_cu, "v_y");
    auto tau_xx = Variable(mesh.get(), initial_zero, boundary_copy(initial_zero), boundary_copy_cu(initial_zero), "tau_xx");
    auto tau_xy = Variable(mesh.get(), initial_zero, boundary_copy(initial_zero), boundary_copy_cu(initial_zero), "tau_xy");
    auto tau_yy = Variable(mesh.get(), initial_zero, boundary_copy(initial_zero), boundary_copy_cu(initial_zero), "tau_yy");

    auto rho = 2650;
    auto lambda = 57.92;
    auto mu = 48.9;

    std::vector<Variable*> all_vars {&v_x, &v_y, &tau_xx, &tau_xy, &tau_yy};
    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu, 0.5, all_vars);

    std::vector<std::tuple<Variable*, char, Variable>> equation_system = {
            {d1t(v_x),    '=', (d1dx(tau_xx) + d1dy(tau_xy)) / rho},
            {d1t(v_y),    '=', (d1dx(tau_xy) + d1dy(tau_yy)) / rho},
            {d1t(tau_xx), '=', (lambda + 2 * mu) * d1dx(v_x) + lambda * d1dy(v_y)},
            {d1t(tau_xy), '=', mu * (d1dx(v_y) + d1dy(v_x))},
            {d1t(tau_yy), '=', (lambda + 2 * mu) * d1dy(v_y) + lambda * d1dx(v_x)},
    };

    auto equation = Equation(timesteps);

    auto store_vars = {&v_x, &v_y};
    initializer.init_store(store_vars);

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, initializer.visualize, store_vars);
    auto end = std::chrono::steady_clock::now();
    if (CFDArcoGlobalInit::get_rank() == 0) std::cout << std::endl << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]" << std::endl;

    initializer.finalize();

    return 0;
}
