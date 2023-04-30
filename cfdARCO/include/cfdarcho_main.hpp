#ifndef CFDARCO_CFDARCHO_MAIN_HPP
#define CFDARCO_CFDARCHO_MAIN_HPP

#include <vector>
#include <mesh2d.hpp>
#include <filesystem>

#include "decls.hpp"

namespace fs = std::filesystem;

enum class DistributionStrategy {Linear, Cluster};

class CFDArcoGlobalInit {
public:
    static void initialize(int argc, char** argv, bool skip_history_, const fs::path& store_path = "./dumps");
    static void finalize();
    static void make_node_distribution(Mesh2D* _mesh, DistributionStrategy distribution_strategy,
                                       std::vector<size_t> priorities = {});
    static std::vector<MatrixX4dRB> get_redistributed(const MatrixX4dRB& inst, const std::string& name);
    static MatrixX4dRB recombine(const MatrixX4dRB& inst, const std::string& name);
    static inline int get_rank() { return world_rank; }
    static void enable_cuda(Mesh2D* mesh, int cuda_ranks);
    static bool cuda_enabled;
    static bool skip_history;
    static bool store_stepping;
    static fs::path store_dir;

    CFDArcoGlobalInit(CFDArcoGlobalInit &other) = delete;
    void operator=(const CFDArcoGlobalInit &) = delete;


private:
    ~CFDArcoGlobalInit();
    static std::vector<std::vector<size_t>> get_send_perspective(std::vector<size_t>& proc_node_distribution,
                                                                 Mesh2D* mesh, size_t proc_rank);

    static std::vector<std::vector<size_t>> node_distribution;
    static std::vector<size_t> current_proc_node_distribution;
    static std::vector<int> node_id_to_proc;
    static std::vector<std::vector<size_t>> current_proc_node_receive_distribution;
    static std::vector<std::vector<size_t>> current_proc_node_send_distribution;
    static Mesh2D* mesh;
    static int world_size;
    static int world_rank;
};


#endif //CFDARCO_CFDARCHO_MAIN_HPP
