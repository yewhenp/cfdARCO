#ifndef CFDARCO_CFDARCHO_MAIN_HPP
#define CFDARCO_CFDARCHO_MAIN_HPP

#include <vector>
#include <mesh2d.hpp>
#include "decls.hpp"


class CFDArcoGlobalInit {
public:
    static void initialize(int argc, char** argv);
    static void finalize();
    static void make_node_distribution(Mesh2D* _mesh);
    static std::vector<MatrixX4dRB> get_redistributed(const MatrixX4dRB& inst, const std::string& name);
    static MatrixX4dRB recombine(const MatrixX4dRB& inst, const std::string& name);
    static inline int get_rank() { return world_rank; }

    CFDArcoGlobalInit(CFDArcoGlobalInit &other) = delete;
    void operator=(const CFDArcoGlobalInit &) = delete;

private:
    ~CFDArcoGlobalInit();

    static std::vector<std::vector<size_t>> node_distribution;
    static std::vector<size_t> current_proc_node_distribution;
    static std::vector<size_t> nums_nodes_per_proc;
    static Mesh2D* mesh;
    static int world_size;
    static int world_rank;
    static int num_modes_per_proc;
};


#endif //CFDARCO_CFDARCHO_MAIN_HPP
