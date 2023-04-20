#ifndef CFDARCO_DISTRIBUTION_ALGO_HPP
#define CFDARCO_DISTRIBUTION_ALGO_HPP

#include "mesh2d.hpp"

std::vector<int> linear_distribution(Mesh2D* mesh, size_t num_proc, std::vector<size_t>& priorities);
std::vector<int> cluster_distribution(Mesh2D* mesh, size_t num_proc, std::vector<size_t>& priorities);

#endif //CFDARCO_DISTRIBUTION_ALGO_HPP
