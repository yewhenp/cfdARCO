#include "io_operators.hpp"
#include <chrono>
#include <ctime>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

void store_history(const std::vector<Variable*>& vars_to_store, const Mesh2D* mesh, const fs::path& store_path) {
    std::time_t t = std::time(0);
    std::tm* now = std::localtime(&t);
    std::string timestamp = std::to_string(now->tm_year + 1900) + "_" +  std::to_string(now->tm_mon + 1) + "_" +
            std::to_string(now->tm_mday) + "_" + std::to_string(now->tm_hour) + "_" + std::to_string(now->tm_min) +
            "_" + std::to_string(now->tm_sec);

    auto store_dir = store_path / ("run_" + timestamp);
    fs::create_directories(store_dir);

    for(auto var : vars_to_store) {

        auto store_var_dir = store_dir / var->name;
        fs::create_directories(store_var_dir);

        for (int i = 0; i < var->history.size() - 1; ++i) {
            std::fstream file;
            file.open(store_var_dir / (std::to_string(i) + ".bin"), std::ios_base::out | std::ios_base::binary);

            if(!file.is_open())
            {
                std::cerr << "Unable to open the file" << std::endl;
                return;
            }

            auto grid_hist = to_grid(mesh, var->history[i]);
            file.write(reinterpret_cast<char*>(grid_hist.data()), grid_hist.rows() * grid_hist.cols() * sizeof(double));
            file.close();
        }
    }

    store_mesh(mesh, store_dir);

    auto store_dir_latest = store_path / "run_latest";
    fs::remove_all(store_dir_latest);
    const auto copy_options = fs::copy_options::update_existing | fs::copy_options::recursive;
    fs::copy(store_dir, store_dir_latest, copy_options);


}


void store_mesh(const Mesh2D* mesh, const fs::path& store_path) {
    json mesh_json;

    auto vertexes = json::array();
    auto edges = json::array();
    auto nodes = json::array();

    for (auto vrtx : mesh->_vertexes) {
        vertexes.push_back(json::array({vrtx->x(), vrtx->y()}));
    }
    for (auto edge : mesh->_edges) {
        auto vertexes_id = json::array({edge->_vertexes_id.at(0), edge->_vertexes_id.at(1)});
        auto nodes_id = json::array();
        for (auto node_id : edge->_nodes_id) {
            nodes_id.push_back(node_id);
        }
        edges.push_back({
            {"vertexes_id", vertexes_id},
            {"nodes_id", nodes_id},
        });
    }
    for (auto node : mesh->_nodes) {
        auto node_edges_arr = json::array();
        for (auto edge_id : node->_edges_id) {
            node_edges_arr.push_back(edge_id);
        }
        auto node_vrtx_arr = json::array();
        for (auto vrtx_id : node->_vertexes_id) {
            node_vrtx_arr.push_back(vrtx_id);
        }
        nodes.push_back({
            {"vertexes", node_vrtx_arr},
            {"edges", node_edges_arr},
        });
    }

    mesh_json["vertexes"] = vertexes;
    mesh_json["edges"] = edges;
    mesh_json["nodes"] = nodes;

    mesh_json["x"] = mesh->_x;
    mesh_json["y"] = mesh->_y;
    mesh_json["lx"] = mesh->_lx;
    mesh_json["ly"] = mesh->_ly;

    std::ofstream o(store_path / "mesh.json");
    o << std::setw(4) << mesh_json << std::endl;
}


std::shared_ptr<Mesh2D> read_mesh(const fs::path& store_path) {
    std::ifstream f(store_path);
    json data = json::parse(f);

    auto mesh = std::make_shared<Mesh2D>(data["x"], data["y"], data["lx"], data["ly"]);
    for (int i = 0; i < data["vertexes"].size(); ++i) {
        auto vrtx = std::make_shared<Vertex2D>(data["vertexes"][i][0], data["vertexes"][i][1], i);
        mesh->_vertexes.push_back(vrtx);
    }
    for (int i = 0; i < data["edges"].size(); ++i) {
        auto edge_json = data["edges"][i];
        auto edge = std::make_shared<Edge2D>(edge_json["vertexes_id"][0], edge_json["vertexes_id"][1], i);
        for (auto node_id : edge_json["nodes_id"]) {
            edge->_nodes_id.push_back(node_id);
        }
        mesh->_edges.push_back(edge);
    }
    for (int i = 0; i < data["nodes"].size(); ++i) {
        auto node_edges_arr = data["nodes"][i]["edges"];
        auto node_vrtx_arr = data["nodes"][i]["vertexes"];
        auto node = std::make_shared<Quadrangle2D>( node_edges_arr[0], node_edges_arr[1], node_edges_arr[2], node_edges_arr[3],
                                        node_vrtx_arr[0], node_vrtx_arr[1], node_vrtx_arr[2], node_vrtx_arr[3], i);

        mesh->_nodes.push_back(node);
    }

    mesh->compute();

    auto mesh2 = Mesh2D{data["x"], data["y"], data["lx"], data["ly"]};
    mesh2.init_basic_internals();
    mesh2.compute();

    if (!mesh2._vec_in_edge_neigh_direction_x.isApprox(mesh->_vec_in_edge_neigh_direction_x)) std::cerr << "mesh2._vec_in_edge_neigh_direction_x" << std::endl;
    if (!mesh2._vec_in_edge_neigh_direction_y.isApprox(mesh->_vec_in_edge_neigh_direction_y)) std::cerr << "mesh2._vec_in_edge_neigh_direction_y" << std::endl;
    if (!mesh2._vec_in_edge_direction_x.isApprox(mesh->_vec_in_edge_direction_x)) std::cerr << "mesh2._vec_in_edge_direction_x" << std::endl;
    if (!mesh2._vec_in_edge_direction_y.isApprox(mesh->_vec_in_edge_direction_y)) std::cerr << "mesh2._vec_in_edge_direction_y" << std::endl;
    if (!mesh2._volumes.isApprox(mesh->_volumes)) std::cerr << "mesh2._volumes" << std::endl;

    return mesh;
}


std::pair<std::shared_ptr<Mesh2D>, std::vector<Variable>> read_history(const fs::path& store_path) {
    auto mesh = read_mesh(store_path / "mesh.json");
    std::vector<Variable> vars;

    for (const auto & entry : fs::directory_iterator(store_path)) {
        if (entry.is_directory()) {
            Eigen::VectorXd initial = Eigen::VectorXd::Zero(mesh->_num_nodes);
            auto var_name = entry.path().filename().string();
            BoundaryFN bound = [] (Mesh2D* mesh, Eigen::VectorXd& arr) { throw std::runtime_error{"Using uninitialized Variable"}; return Eigen::VectorXd{}; };
            auto var = Variable(mesh.get(),
                                initial,
                                bound,
                                var_name);

            auto num_entries = std::distance(fs::directory_iterator(entry), fs::directory_iterator{});
            for (int i = 0; i < num_entries; ++i) {
                std::ifstream file( entry.path() / (std::to_string(i) + ".bin"), std::ios::binary );
                if(!file.is_open())
                {
                    throw std::runtime_error{"Unable to open the file"};
                }

                MatrixX4dRB grid {mesh->_x, mesh->_y};
                file.read(reinterpret_cast<char*>(grid.data()), grid.size() * sizeof(double));
                file.close();

                if(!file.good()) {
                    throw std::runtime_error{"Error occurred at reading time!"};
                }

                auto hist_var = from_grid(mesh.get(), grid);
                var.history.push_back(hist_var);
            }

            vars.push_back(var);
        }
    }

    return {mesh, vars};
}