#include <mpi.h>
#include "cfdarcho_main.hpp"

CFDArcoGlobalInit::CFDArcoGlobalInit(int argc, char **argv) {
    world_size = 0;
    world_rank = 0;
    mesh = nullptr;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
}

CFDArcoGlobalInit::~CFDArcoGlobalInit() {
    MPI_Finalize();
}

void CFDArcoGlobalInit::make_node_distribution(Mesh2D *_mesh) {
    mesh = _mesh;
    size_t num_modes_per_proc = (mesh->_num_nodes / world_size) + 1;

    size_t current_proc = -1;
    for (size_t i = 0; i < mesh->_num_nodes; ++i) {
        if (i % num_modes_per_proc == 0) {
            current_proc += 1;
            node_distribution.emplace_back();
        }
        node_distribution[current_proc].push_back(i);
    }
}

CFDArcoGlobalInit* CFDArcoGlobalInit::initialize(int argc, char **argv) {
    if (singleton_ == nullptr) {
        singleton_ = new CFDArcoGlobalInit(argc, argv);
        return singleton_;
    }
    throw std::invalid_argument("This should be called only once");
}




