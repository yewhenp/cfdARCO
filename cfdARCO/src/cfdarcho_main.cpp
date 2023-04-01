// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <Eigen/Core>
#include <mpi.h>
#include "cfdarcho_main.hpp"

std::vector<std::vector<size_t>> CFDArcoGlobalInit::node_distribution = {};
std::vector<size_t> CFDArcoGlobalInit::current_proc_node_distribution = {};
std::vector<size_t> CFDArcoGlobalInit::nums_nodes_per_proc = {};
Mesh2D* CFDArcoGlobalInit::mesh = nullptr;
int CFDArcoGlobalInit::world_size = 0;
int CFDArcoGlobalInit::world_rank = 0;
int CFDArcoGlobalInit::num_modes_per_proc = 0;

void CFDArcoGlobalInit::finalize() {
    MPI_Finalize();
}

void CFDArcoGlobalInit::make_node_distribution(Mesh2D *_mesh) {
    mesh = _mesh;
    num_modes_per_proc = (mesh->_num_nodes / world_size) + 1;

    size_t current_proc = -1;
    size_t current_proc_num = 0;
    bool flag = false;
    for (size_t i = 0; i < mesh->_num_nodes; ++i) {
        if (i % num_modes_per_proc == 0) {
            current_proc += 1;
            node_distribution.emplace_back();
            if (flag)
                nums_nodes_per_proc.push_back(current_proc_num);
            current_proc_num = 0;
            flag = true;
        }
        node_distribution[current_proc].push_back(mesh->_nodes[i]->_id);
        current_proc_num += 1;
    }
    if (current_proc_num > 0)
        nums_nodes_per_proc.push_back(current_proc_num);

    for (auto& node_list : node_distribution) {
        size_t lastt = node_list[0];
        for (int i = 1; i < node_list.size(); ++i) {
            if(lastt != node_list[i] - 1) {
                std::cerr << "NOT contin" << std::endl;
                exit(-4);
            }
            lastt = node_list[i];
        }
    }

    current_proc_node_distribution = node_distribution[world_rank];

    mesh->_volumes = mesh->_volumes_tot(current_proc_node_distribution, Eigen::all);
    mesh->_normal_x = mesh->_normal_x_tot(current_proc_node_distribution, Eigen::all);
    mesh->_normal_y = mesh->_normal_y_tot(current_proc_node_distribution, Eigen::all);
    mesh->_vec_in_edge_direction_x = mesh->_vec_in_edge_direction_x_tot(current_proc_node_distribution, Eigen::all);
    mesh->_vec_in_edge_direction_y = mesh->_vec_in_edge_direction_y_tot(current_proc_node_distribution, Eigen::all);
    mesh->_vec_in_edge_neigh_direction_x = mesh->_vec_in_edge_neigh_direction_x_tot(current_proc_node_distribution, Eigen::all);
    mesh->_vec_in_edge_neigh_direction_y = mesh->_vec_in_edge_neigh_direction_y_tot(current_proc_node_distribution, Eigen::all);
    mesh->_n2_ids = Eigen::MatrixX4d {current_proc_node_distribution.size(), 4};
    for (int i = 0; i < 4; ++i) {
        int qq = 0;
        for (auto current_proc_node : current_proc_node_distribution) {
            mesh->_n2_ids(qq, i) = mesh->_n2_ids_tot(current_proc_node, i);
            qq++;
        }
    }
    mesh->_num_nodes = current_proc_node_distribution.size();
    mesh->_nodes_tot = mesh->_nodes;
    mesh->_nodes = {};
    for (auto node_id : current_proc_node_distribution) {
        mesh->_nodes.push_back(mesh->_nodes_tot[node_id]);
    }
}

std::vector<MatrixX4dRB> CFDArcoGlobalInit::get_redistributed(const MatrixX4dRB& inst, const std::string& name) {
    MatrixX4dRB buff {mesh->_num_nodes_tot, inst.cols()};
    buff.setConstant(0);
    std::vector<int> send_sizes;
    std::vector<int> send_displ;
    int curr_displ = 0;
    for (auto num_nodes : nums_nodes_per_proc) {
        send_sizes.push_back(num_nodes * inst.cols());
        send_displ.push_back(curr_displ);
        curr_displ += num_nodes * inst.cols();
    }
    MPI_Allgatherv(inst.data(), inst.rows() * inst.cols(), MPI_DOUBLE, buff.data(), send_sizes.data(), send_displ.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    std::vector<MatrixX4dRB> ret = {};
//    ret.reserve(4);
    for (int idx = 0; idx < mesh->_n2_ids.cols(); ++idx) {
        ret.emplace_back(buff(mesh->_n2_ids.col(idx), Eigen::all));
    }

    return ret;
}

MatrixX4dRB CFDArcoGlobalInit::recombine(const MatrixX4dRB& inst, const std::string& name) {
    MatrixX4dRB buff {mesh->_num_nodes_tot, inst.cols()};
    buff.setConstant(0);
    std::vector<int> send_sizes;
    std::vector<int> send_displ;
    int curr_displ = 0;
    for (auto num_nodes : nums_nodes_per_proc) {
        send_sizes.push_back(num_nodes * inst.cols());
        send_displ.push_back(curr_displ);
        curr_displ += num_nodes * inst.cols();
    }
    MPI_Allgatherv(inst.data(), inst.rows() * inst.cols(), MPI_DOUBLE, buff.data(), send_sizes.data(), send_displ.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    return buff;
}

void CFDArcoGlobalInit::initialize(int argc, char **argv) {
    world_size = 0;
    world_rank = 0;
    num_modes_per_proc = 0;
    mesh = nullptr;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
}




