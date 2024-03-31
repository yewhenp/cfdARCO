
#ifndef CFDARCO_VAL_UTILS_HPP
#define CFDARCO_VAL_UTILS_HPP

#include "mesh2d.hpp"
#include "cfdarcho_main.hpp"

Eigen::VectorXd initial_with_val(Mesh2D* mesh, double val);

Eigen::VectorXd _boundary_copy(Mesh2D* mesh, Eigen::VectorXd& arr, const Eigen::VectorXd& copy_var);

CudaDataMatrix _boundary_copy_cu(Mesh2D* mesh, CudaDataMatrix& arr, const CudaDataMatrix& copy_var);

inline auto boundary_copy(const Eigen::VectorXd& copy_var) {
    return [copy_var] (Mesh2D* mesh, Eigen::VectorXd& arr) { return _boundary_copy(mesh, arr, copy_var); };
}

inline auto boundary_copy_cu(const Eigen::VectorXd& copy_var) {
    CudaDataMatrix cuda_copy_var;
    if (CFDArcoGlobalInit::cuda_enabled) {
        cuda_copy_var = CudaDataMatrix::from_eigen(copy_var);
    }

    return [cuda_copy_var](Mesh2D *mesh, CudaDataMatrix &arr) {
        if (CFDArcoGlobalInit::cuda_enabled)
            return _boundary_copy_cu(mesh, arr, cuda_copy_var);
        else
            return cuda_copy_var;
    };
}


Eigen::VectorXd _boundary_neumann(Mesh2D* mesh, Eigen::VectorXd& arr, const Eigen::VectorXd& grad_var);

inline auto boundary_neumann(const Eigen::VectorXd& grad_var) {
    return [grad_var] (Mesh2D* mesh, Eigen::VectorXd& arr) { return _boundary_neumann(mesh, arr, grad_var); };
}

inline Eigen::VectorXd boundary_none(Mesh2D* mesh, Eigen::VectorXd& arr) {
    return arr;
}

inline CudaDataMatrix boundary_none_cu(Mesh2D* mesh, CudaDataMatrix& arr) {
    return arr;
}

#endif //CFDARCO_VAL_UTILS_HPP
