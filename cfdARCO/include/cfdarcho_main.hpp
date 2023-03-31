#ifndef CFDARCO_CFDARCHO_MAIN_HPP
#define CFDARCO_CFDARCHO_MAIN_HPP

#include <vector>
#include <mesh2d.hpp>


class CFDArcoGlobalInit {
public:
    static CFDArcoGlobalInit* initialize(int argc, char** argv);
    void make_node_distribution(Mesh2D* _mesh);

    CFDArcoGlobalInit(CFDArcoGlobalInit &other) = delete;
    void operator=(const CFDArcoGlobalInit &) = delete;

private:
    CFDArcoGlobalInit(int argc, char** argv);
    ~CFDArcoGlobalInit();

    static CFDArcoGlobalInit* singleton_;

    std::vector<std::vector<size_t>> node_distribution;
    Mesh2D* mesh;
    int world_size;
    int world_rank;
};


#endif //CFDARCO_CFDARCHO_MAIN_HPP
