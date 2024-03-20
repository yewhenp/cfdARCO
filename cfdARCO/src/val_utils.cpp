#include <val_utils.hpp>
#include "cfdarcho_main.hpp"

Eigen::VectorXd initial_with_val(Mesh2D* mesh, double val) {
    auto ret = Eigen::VectorXd{mesh->_num_nodes};
    int i = 0;
    for (auto& node : mesh->_nodes) {
        ret(i) = val;
        ++i;
    }
    return ret;
}

Eigen::VectorXd _boundary_copy(Mesh2D* mesh, Eigen::VectorXd& arr, const Eigen::VectorXd& copy_var) {
    auto arr1 = arr.cwiseProduct(mesh->_node_is_boundary_reverce);
    auto copy_var1 = copy_var.cwiseProduct(mesh->_node_is_boundary);
    return arr1 + copy_var1;
}

CudaDataMatrix _boundary_copy_cu(Mesh2D* mesh, CudaDataMatrix& arr, const CudaDataMatrix& copy_var) {
    auto arr1 = arr * mesh->_node_is_boundary_reverce_cu;
    auto copy_var1 = copy_var * mesh->_node_is_boundary_cu;
    return arr1 + copy_var1;
}


Eigen::VectorXd _boundary_neumann(Mesh2D* mesh, Eigen::VectorXd& arr, const Eigen::VectorXd& grad_var) {
    auto redist = CFDArcoGlobalInit::get_redistributed(arr, "boundary_with_neumann");
    auto ret = Eigen::VectorXd{mesh->_num_nodes};
    int i = 0;
    for (auto& node : mesh->_nodes) {
        if (node->is_boundary()) {
            int q = 0;
            MatrixX4dRB is_bound{1, 4};
            is_bound.setConstant(0.0);
            MatrixX4dRB is_not_bound{1, 4};
            is_not_bound.setConstant(1.0);

            for (auto edge_id : node->_edges_id) {
                auto edge = mesh->_edges.at(edge_id);
                if (edge->is_boundary()) {
                    is_bound(q) = 1.0;
                    is_not_bound(q) = 0.0;
                }
                q++;
            }

            auto grad_cur_val = grad_var(i);
            auto nominator = 2*grad_cur_val*node->_volume;
            for (int j = 0; j < 4; ++j) {
                nominator = nominator + is_not_bound(j) * redist.at(j)(i) * mesh->_normal_y(i, j);
            }

            auto ghost_normals = mesh->_normal_y.block<1, 4>(i, 0).cwiseProduct(is_bound);
            auto demon = mesh->_normal_y.row(i).sum() + ghost_normals.sum();
            ret(i) = nominator / demon;

        } else {
            ret(i) = arr(i);
        }
        ++i;
    }
    return ret;
}

