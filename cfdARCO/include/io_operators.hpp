#ifndef CFDARCO_IO_OPERATORS_HPP
#define CFDARCO_IO_OPERATORS_HPP

#include "mesh2d.hpp"
#include "fvm.hpp"

#include <filesystem>

namespace fs = std::filesystem;

void store_history(const std::vector<Variable*>& vars_to_store, const Mesh2D* mesh, const fs::path& store_path = "./dumps");
void store_mesh(const Mesh2D* mesh, const fs::path& store_path = "./dumps");

std::pair<std::shared_ptr<Mesh2D>, std::vector<Variable>> read_history(const fs::path& store_path = "./dumps/run_latest/");
std::shared_ptr<Mesh2D> read_mesh(const fs::path& store_path = "./dumps/run_latest/mesh.json");

#endif //CFDARCO_IO_OPERATORS_HPP
