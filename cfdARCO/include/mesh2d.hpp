#ifndef CFDARCO_MESH2D_HPP
#define CFDARCO_MESH2D_HPP

#include "abstract_mesh.hpp"
#include <unordered_map>
#include <memory>


class Vertex2D;
class Edge2D;
class Quadrangle2D;
class Mesh2D;

using PtrVertex2D = std::shared_ptr<Vertex2D>;
using PtrEdge2D = std::shared_ptr<Edge2D>;
using PtrQuadrangle2D = std::shared_ptr<Quadrangle2D>;


class Vertex2D : AbstractVertex {
public:
    Vertex2D(double x, double y, size_t id) : _id{id}, _coords(x, y) {};
    void compute() override {};
    [[nodiscard]] Eigen::VectorXd coordinates() const override;
    [[nodiscard]] double x() const;
    [[nodiscard]] double y() const;

    Mesh2D* _mesh = nullptr;
    size_t _id;
    Eigen::Vector2d _coords;
};

class Edge2D : AbstractEdge {
public:
    Edge2D(size_t vrtx_1_id, size_t vrtx_2_id, size_t id) : _id{id}, _vertexes_id{vrtx_1_id, vrtx_2_id} {};
    void compute() override;
    [[nodiscard]] bool is_boundary() const override;

    Mesh2D* _mesh = nullptr;
    size_t _id;
    std::array<size_t, 2> _vertexes_id;
    std::vector<size_t> _nodes_id {};
    Eigen::Vector2d _normal = {0, 0};
    Eigen::Vector2d _center_coords = {0, 0};
    double _area = 0;
};

class Quadrangle2D : AbstractCell {
public:
    Quadrangle2D(size_t e1, size_t e2, size_t e3, size_t e4, size_t v1, size_t v2, size_t v3, size_t v4, size_t id) :
                 _id{id}, _edges_id{e1, e2, e3, e4}, _vertexes_id{v1, v2, v3, v4} {};
    void compute() override;
    [[nodiscard]] Eigen::VectorXd center_coords() const override;
    [[nodiscard]] bool is_boundary() const override;
    [[nodiscard]] double x() const;
    [[nodiscard]] double y() const;

    Mesh2D* _mesh = nullptr;
    size_t _id;
    std::array<size_t, 4> _edges_id;
    std::array<size_t, 4> _vertexes_id;
    Eigen::Vector2d _center_coords = {0, 0};
    Eigen::Matrix<double, 4, 2> _vectors_in_edges_directions = {};
    std::unordered_map<size_t, Eigen::Matrix<double, 1, 2>> _vectors_in_edges_directions_by_id = {};
    Eigen::Matrix<double, 4, 2> _normals = {};
    double _volume = 0;
};

class Mesh2D : AbstractMesh {
public:
    Mesh2D(size_t x, size_t y, double lx, double ly) : _num_nodes{x*y}, _x{x}, _y{y},
            _lx{lx}, _ly{ly}, _dx{lx / static_cast<double>(x)}, _dy{ly / static_cast<double>(y)} {};
    void compute() override;
    void init_basic_internals();
    [[nodiscard]] size_t coord_fo_idx(size_t x, size_t y) const;

    size_t _num_nodes;
    size_t _x;
    size_t _y;
    double _lx;
    double _ly;
    double _dx;
    double _dy;
    std::vector<std::shared_ptr<Vertex2D>> _vertexes{};
    std::vector<std::shared_ptr<Edge2D>> _edges{};
    std::vector<std::shared_ptr<Quadrangle2D>> _nodes{};
    Eigen::VectorXd _volumes{};
    Eigen::MatrixX4d _normal_x{};
    Eigen::MatrixX4d _normal_y{};
};


#endif //CFDARCO_MESH2D_HPP
