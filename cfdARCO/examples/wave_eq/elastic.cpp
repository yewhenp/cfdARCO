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

    auto rho = 2800;
    auto A = 10e7;
    auto f_0 = 200;
    auto v_p = 3300;
    auto v_s = 1905;
    auto mu = (v_s * v_s) * rho;
    auto lambda = (v_p * v_p) * rho - 2*mu;

    auto initial_zero = initial_with_val(mesh.get(), 0);
    auto initial_vx = initial_with_val(mesh.get(), 0);
    auto initial_vy = initial_with_val(mesh.get(), 0);

    auto v_x = Variable(mesh.get(), initial_vx, boundary_copy(initial_vx), boundary_copy_cu(initial_zero), "v_x");
    auto v_y = Variable(mesh.get(), initial_vy, boundary_copy(initial_vy), boundary_copy_cu(initial_zero), "v_y");
    auto tau_xx = Variable(mesh.get(), initial_zero, boundary_copy(initial_zero), boundary_copy_cu(initial_zero), "tau_xx");
    auto tau_xy = Variable(mesh.get(), initial_zero, boundary_copy(initial_zero), boundary_copy_cu(initial_zero), "tau_xy");
    auto tau_yy = Variable(mesh.get(), initial_zero, boundary_copy(initial_zero), boundary_copy_cu(initial_zero), "tau_yy");

    std::vector<Variable*> all_vars {&v_x, &v_y, &tau_xx, &tau_xy, &tau_yy};
    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu, 0.05, all_vars);


    auto central_point = Variable(mesh.get(), 0.0);
    sett(central_point.current, mesh->_x, mesh->_y, mesh->_x * 0.25, mesh->_y * 0.25, 1.0);
    sett(central_point.current, mesh->_x, mesh->_y, mesh->_x * 0.75, mesh->_y * 0.75, 1.0);
    auto curr_time = PointerVariable(mesh.get(), &dt._current_time_dbl);

    auto a = 3.14 * 3.14 * f_0 * f_0;
    auto s_base = -2 * A * a * curr_time * exp(-a * curr_time * curr_time);
    auto s = s_base * central_point;

    std::vector<std::tuple<Variable*, char, Variable>> equation_system = {
            {d1t(v_x),    '=', (d1dx(tau_xx) + d1dy(tau_xy) + s) / rho},
            {d1t(v_y),    '=', (d1dx(tau_xy) + d1dy(tau_yy) + s) / rho},
            {d1t(tau_xx), '=', (lambda + 2 * mu) * d1dx(v_x) + lambda * d1dy(v_y)},
            {d1t(tau_xy), '=', mu * (d1dx(v_y) + d1dy(v_x))},
            {d1t(tau_yy), '=', (lambda + 2 * mu) * d1dy(v_y) + lambda * d1dx(v_x)},
    };

    auto equation = Equation(timesteps);

    auto store_vars = {&v_x, &v_y, &tau_xx, &tau_xy, &tau_yy};
    initializer.init_store(store_vars);

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, initializer.visualize, store_vars);
    auto end = std::chrono::steady_clock::now();
    if (CFDArcoGlobalInit::get_rank() == 0) std::cout << std::endl << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]" << std::endl;

    initializer.finalize();

    return 0;
}
