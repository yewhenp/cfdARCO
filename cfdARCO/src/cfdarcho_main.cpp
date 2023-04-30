// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <Eigen/Core>
#include <mpi.h>
#include <numeric>
#include <algorithm>
#include <random>
#include "cfdarcho_main.hpp"
#include "distribution_algo.hpp"
#include "cuda_operators.hpp"

std::vector<std::vector<size_t>> CFDArcoGlobalInit::node_distribution = {};
std::vector<size_t> CFDArcoGlobalInit::current_proc_node_distribution = {};
std::vector<int> CFDArcoGlobalInit::node_id_to_proc = {};
std::vector<std::vector<size_t>> CFDArcoGlobalInit::current_proc_node_send_distribution = {};
std::vector<std::vector<size_t>> CFDArcoGlobalInit::current_proc_node_receive_distribution = {};
Mesh2D* CFDArcoGlobalInit::mesh = nullptr;
int CFDArcoGlobalInit::world_size = 0;
int CFDArcoGlobalInit::world_rank = 0;
bool CFDArcoGlobalInit::cuda_enabled = false;

void CFDArcoGlobalInit::finalize() {
    MPI_Finalize();

#ifdef CFDARCHO_CUDA_ENABLE
    if (cuda_enabled) {
        Allocator::cuda_mem_pool.reset();
        Allocator::allocator_alive = false;
    }
#endif
}


std::vector<std::vector<size_t>> CFDArcoGlobalInit::get_send_perspective(std::vector<size_t>& proc_node_distribution,
                                                      Mesh2D* mesh, size_t proc_rank){
    std::vector<std::vector<size_t>> ret{};
    for (int i = 0; i < CFDArcoGlobalInit::world_size; ++i) {
        ret.emplace_back();
    }
    for (auto node_id : proc_node_distribution) {
        for (auto neigh_id : mesh->get_ids_of_neightbours(node_id)) {
            if (node_id_to_proc[neigh_id] != proc_rank) {
                auto proc_id = node_id_to_proc[neigh_id];
                if (!std::count(ret[proc_id].begin(),
                                ret[proc_id].end(), node_id)) {
                    ret[proc_id].push_back(node_id);
                }
            }
        }
    }
    return ret;
}

void CFDArcoGlobalInit::make_node_distribution(Mesh2D *_mesh, DistributionStrategy distribution_strategy, std::vector<size_t> priorities) {
    mesh = _mesh;

    node_id_to_proc = std::vector<int>(mesh->_num_nodes);
    if (world_rank == 0) {
        if (priorities.size() != world_size) {
            priorities = std::vector<size_t>(world_size, 1);
            std::cout << "Using default priorities" << std::endl;
        }
        std::cout << "priorities: ";
        for (auto pr : priorities) {
            std::cout << pr << " ";
        }
        std::cout << std::endl;

        if (distribution_strategy == DistributionStrategy::Cluster)
            node_id_to_proc = cluster_distribution(mesh, world_size, priorities);
        if (distribution_strategy == DistributionStrategy::Linear)
            node_id_to_proc = linear_distribution(mesh, world_size, priorities);
    }
    MPI_Bcast(node_id_to_proc.data(), mesh->_num_nodes, MPI_INT, 0, MPI_COMM_WORLD);

    node_distribution = std::vector<std::vector<size_t>>(world_size);
    for (int i = 0; i < mesh->_num_nodes; ++i) {
        node_distribution[node_id_to_proc[i]].push_back(i);
    }

    if (world_rank == 0) {
        std::cout << "Cluster sizes: ";
        for (int i = 0; i < world_size; ++i) {
            std::cout << node_distribution.at(i).size() << " ";
        }
        std::cout << std::endl;
    }

    current_proc_node_distribution = node_distribution[world_rank];

    std::vector<std::vector<std::vector<size_t>>> sending_nodes_from_proc_perspectiv = {};
    for (int i = 0; i < world_size; ++i) {
        sending_nodes_from_proc_perspectiv.push_back(get_send_perspective(node_distribution[i], mesh, i));
    }

    current_proc_node_send_distribution = sending_nodes_from_proc_perspectiv[world_rank];
    current_proc_node_receive_distribution = {};
    for (int i = 0; i < world_size; ++i) {
        current_proc_node_receive_distribution.push_back(sending_nodes_from_proc_perspectiv[i][world_rank]);
    }


    mesh->_volumes = mesh->_volumes_tot(current_proc_node_distribution, Eigen::placeholders::all);
    mesh->_normal_x = mesh->_normal_x_tot(current_proc_node_distribution, Eigen::placeholders::all);
    mesh->_normal_y = mesh->_normal_y_tot(current_proc_node_distribution, Eigen::placeholders::all);
    mesh->_vec_in_edge_direction_x = mesh->_vec_in_edge_direction_x_tot(current_proc_node_distribution, Eigen::placeholders::all);
    mesh->_vec_in_edge_direction_y = mesh->_vec_in_edge_direction_y_tot(current_proc_node_distribution, Eigen::placeholders::all);
    mesh->_vec_in_edge_neigh_direction_x = mesh->_vec_in_edge_neigh_direction_x_tot(current_proc_node_distribution, Eigen::placeholders::all);
    mesh->_vec_in_edge_neigh_direction_y = mesh->_vec_in_edge_neigh_direction_y_tot(current_proc_node_distribution, Eigen::placeholders::all);
    mesh->_node_is_boundary = mesh->_node_is_boundary_tot(current_proc_node_distribution, Eigen::placeholders::all);
    mesh->_node_is_boundary_reverce = mesh->_node_is_boundary_reverce_tot(current_proc_node_distribution, Eigen::placeholders::all);
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
    buff(current_proc_node_distribution, Eigen::placeholders::all) = inst;

    std::vector<MatrixX4dRB> input_buffers;
    std::vector<MatrixX4dRB> output_buffers;
    std::vector<MPI_Request> mpi_requests;
    std::vector<MPI_Status> mpi_statuses;
    for (int i = 0; i < world_size; ++i) {
        input_buffers.emplace_back(current_proc_node_receive_distribution[i].size(), inst.cols());
        output_buffers.emplace_back(buff(current_proc_node_send_distribution[i], Eigen::placeholders::all));
    }

    for (int i = 0; i < world_size; ++i) {
        if (i == world_rank)
            continue;
        if (!current_proc_node_receive_distribution[i].empty()) {
            MPI_Request req;
            mpi_requests.push_back(req);
            MPI_Irecv(input_buffers[i].data(), input_buffers[i].rows() * input_buffers[i].cols(), MPI_DOUBLE, i, i * 100 + world_rank * 1000, MPI_COMM_WORLD, &(mpi_requests.at(mpi_requests.size()-1)));
        }
        if (!current_proc_node_send_distribution[i].empty()) {
            MPI_Request req;
            mpi_requests.push_back(req);
            MPI_Isend(output_buffers[i].data(), output_buffers[i].rows() * output_buffers[i].cols(), MPI_DOUBLE, i, i * 1000 + world_rank * 100, MPI_COMM_WORLD, &(mpi_requests.at(mpi_requests.size()-1)));
        }
    }

    MPI_Waitall(mpi_requests.size(), mpi_requests.data(), mpi_statuses.data());

    for (const auto& st : mpi_statuses) {
        if (st.MPI_ERROR != MPI_SUCCESS) {
            std::cerr << "rank " << CFDArcoGlobalInit::get_rank() << " MPI error code: " << st.MPI_ERROR << std::endl;
        }
    }

    for (int i = 0; i < world_size; ++i) {
        if (i == world_rank)
            continue;
        if (!current_proc_node_receive_distribution[i].empty()) {
            buff(current_proc_node_receive_distribution[i], Eigen::placeholders::all) = input_buffers[i];
        }
    }

    std::vector<MatrixX4dRB> ret = {};
    for (int idx = 0; idx < mesh->_n2_ids.cols(); ++idx) {
        ret.emplace_back(buff(mesh->_n2_ids.col(idx), Eigen::placeholders::all));
    }

    return ret;
}


MatrixX4dRB CFDArcoGlobalInit::recombine(const MatrixX4dRB& inst, const std::string& name) {
    MatrixX4dRB buff {mesh->_num_nodes_tot, inst.cols()};
    buff.setConstant(0);
    buff(current_proc_node_distribution, Eigen::placeholders::all) = inst;

    std::vector<MatrixX4dRB> input_buffers;
    std::vector<MPI_Request> mpi_requests;
    std::vector<MPI_Status> mpi_statuses;

    for (int i = 0; i < world_size; ++i) {
        input_buffers.emplace_back(node_distribution[i].size(), inst.cols());
        if (i == world_rank)
            continue;
        MPI_Request req;
        mpi_requests.push_back(req);
        MPI_Isend(inst.data(), inst.rows() * inst.cols(), MPI_DOUBLE, i, i * 1000 + world_rank * 100, MPI_COMM_WORLD, &(mpi_requests.at(mpi_requests.size()-1)));

        MPI_Request req1;
        mpi_requests.push_back(req1);
        MPI_Irecv(input_buffers[i].data(), input_buffers[i].rows() * input_buffers[i].cols(), MPI_DOUBLE, i, i * 100 + world_rank * 1000, MPI_COMM_WORLD, &(mpi_requests.at(mpi_requests.size()-1)));
    }

    MPI_Waitall(mpi_requests.size(), mpi_requests.data(), mpi_statuses.data());

    for (int i = 0; i < world_size; ++i) {
        if (i == world_rank)
            continue;
        buff(node_distribution[i], Eigen::placeholders::all) = input_buffers[i];
    }

    return buff;
}

#ifdef CFDARCHO_CUDA_ENABLE
rmm::mr::cuda_memory_resource* get_cuda_resource() {
    static rmm::mr::cuda_memory_resource res{};
    return &res;
}
#endif

void CFDArcoGlobalInit::initialize(int argc, char **argv) {
    world_size = 0;
    world_rank = 0;
    mesh = nullptr;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
}

void CFDArcoGlobalInit::enable_cuda(Mesh2D* mesh) {

#ifndef CFDARCHO_CUDA_ENABLE
    throw std::runtime_error{"CUDA not available"};
#else

    cuda_enabled = true;
    auto const [free, total] = rmm::detail::available_device_memory();
    Allocator::cuda_mem_pool = std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
                get_cuda_resource(),
                rmm::detail::align_up(static_cast<std::size_t>(static_cast<double>(free) * 0.8),
                                      rmm::detail::CUDA_ALLOCATION_ALIGNMENT),
                rmm::detail::align_up(static_cast<std::size_t>(static_cast<double>(free) * 0.9),
                          rmm::detail::CUDA_ALLOCATION_ALIGNMENT)
            );
    Allocator::allocator_alive = true;

    mesh->_volumes_cu = CudaDataMatrix::from_eigen(mesh->_volumes);
    mesh->_normal_x_cu = CudaDataMatrix::from_eigen(mesh->_normal_x);
    mesh->_normal_y_cu = CudaDataMatrix::from_eigen(mesh->_normal_y);
    mesh->_vec_in_edge_direction_x_cu = CudaDataMatrix::from_eigen(mesh->_vec_in_edge_direction_x);
    mesh->_vec_in_edge_direction_y_cu = CudaDataMatrix::from_eigen(mesh->_vec_in_edge_direction_y);
    mesh->_vec_in_edge_neigh_direction_x_cu = CudaDataMatrix::from_eigen(mesh->_vec_in_edge_neigh_direction_x);
    mesh->_vec_in_edge_neigh_direction_y_cu = CudaDataMatrix::from_eigen(mesh->_vec_in_edge_neigh_direction_y);
    mesh->_n2_ids_cu = CudaDataMatrix::from_eigen(mesh->_n2_ids);

#endif
}






