// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com


#include <iostream>
#include "mesh2d.hpp"

Eigen::VectorXd Vertex2D::coordinates() const {
    return _coords;
}

double Vertex2D::x() const {
    return _coords(0);
}

double Vertex2D::y() const {
    return _coords(1);
}

void Edge2D::compute() {
    auto p1_coords = _mesh->_vertexes.at(_vertexes_id.at(0))->_coords;
    auto p2_coords = _mesh->_vertexes.at(_vertexes_id.at(1))->_coords;
    _center_coords = (p1_coords + p2_coords) / 2;
    _normal = Eigen::Vector2d {};
    _normal << -(p2_coords(1) - p1_coords(1)), (p2_coords(0) - p1_coords(0));
    _normal = _normal / _normal.norm();
    _area = (p2_coords - p1_coords).norm();
}

bool Edge2D::is_boundary() const {
//    return _nodes_id.at(0) == _nodes_id.at(1);
    return _nodes_id.size() == 1 || _nodes_id.at(0) == _nodes_id.at(1);
}

void Quadrangle2D::compute() {
    auto p1_coords = _mesh->_vertexes.at(_vertexes_id.at(0))->_coords;
    auto p2_coords = _mesh->_vertexes.at(_vertexes_id.at(1))->_coords;
    auto p3_coords = _mesh->_vertexes.at(_vertexes_id.at(2))->_coords;
    auto p4_coords = _mesh->_vertexes.at(_vertexes_id.at(3))->_coords;
    _center_coords = (p1_coords + p2_coords + p3_coords + p4_coords) / 4;
    for (int i=0; i < 4; ++i){
        auto edge = _mesh->_edges.at(_edges_id.at(i));
        auto direction_vector = edge->_center_coords - _center_coords;
        _vectors_in_edges_directions.block<1, 2>(i, 0) = direction_vector;
        _vectors_in_edges_directions_by_id[edge->_id] = direction_vector;
    }

    _volume = 0.5 * (
            p1_coords(0) * p2_coords(1) + p2_coords(0) * p3_coords(1) + p3_coords(0) * p4_coords(1) + p4_coords(0) * p1_coords(1)
            -
            (p1_coords(1) * p2_coords(0) + p2_coords(1) * p3_coords(0) + p3_coords(1) * p4_coords(0) + p4_coords(1) * p1_coords(0) )
    );

    for (size_t i = 0; i < _vertexes_id.size() - 1; ++i) {
        auto v0_id = _vertexes_id.at(i);
        auto v1_id = _vertexes_id.at(i+1);

        auto v0 = _mesh->_vertexes.at(v0_id);
        auto v1 = _mesh->_vertexes.at(v1_id);

        auto dx = v1->x() - v0->x();
        auto dy = v1->y() - v0->y();
        _normals.block<1, 2>(static_cast<int>(i), 0) = Eigen::Vector2d {dy, -dx};
    }

    auto v0_id = _vertexes_id.at(_vertexes_id.size() - 1);
    auto v1_id = _vertexes_id.at(0);

    auto v0 = _mesh->_vertexes.at(v0_id);
    auto v1 = _mesh->_vertexes.at(v1_id);

    auto dx = v1->x() - v0->x();
    auto dy = v1->y() - v0->y();
    _normals.block<1, 2>(static_cast<int>(_vertexes_id.size()) - 1, 0) = Eigen::Vector2d {dy, -dx};
}

Eigen::VectorXd Quadrangle2D::center_coords() const {
    return _center_coords;
}

bool Quadrangle2D::is_boundary() const {
    return std::any_of(_edges_id.begin(), _edges_id.end(),
                       [&](size_t edge_id) { return _mesh->_edges.at(edge_id)->is_boundary(); });
}

double Quadrangle2D::x() const {
    return _center_coords(0);
}

double Quadrangle2D::y() const {
    return _center_coords(1);
}


void Mesh2D::compute() {
    for (auto& entry : _vertexes) {
        entry->_mesh = this;
        entry->compute();
    }
    for (auto& entry : _edges) {
        entry->_mesh = this;
        entry->compute();
    }
    for (auto& entry : _nodes) {
        entry->_mesh = this;
        entry->compute();
    }

    _volumes_tot = Eigen::VectorXd {_num_nodes_tot};
    for (const auto& node : _nodes) {
        _volumes_tot(node->_id) = node->_volume;
    }

    for (int i = 0; i < 4; ++i) {
        _n2_ids_tot.emplace_back();
    }

    _normal_x_tot = Eigen::MatrixX4d {_num_nodes_tot, 4};
    _normal_y_tot = Eigen::MatrixX4d {_num_nodes_tot, 4};
    _vec_in_edge_direction_x_tot = Eigen::MatrixX4d {_num_nodes_tot, 4};
    _vec_in_edge_direction_y_tot = Eigen::MatrixX4d {_num_nodes_tot, 4};
    _vec_in_edge_neigh_direction_x_tot = Eigen::MatrixX4d {_num_nodes_tot, 4};
    _vec_in_edge_neigh_direction_y_tot = Eigen::MatrixX4d {_num_nodes_tot, 4};
    for (size_t i = 0; i < _num_nodes_tot; ++i) {
        auto& node = _nodes.at(i);
        Eigen::Vector4d norm_v_x {node->_normals.block<4, 1>(0, 0)};
        Eigen::Vector4d norm_v_y {node->_normals.block<4, 1>(0, 1)};
        _normal_x_tot.block<1, 4>(node->_id, 0) = norm_v_x;
        _normal_y_tot.block<1, 4>(node->_id, 0) = norm_v_y;

        for (int j = 0; j < node->_edges_id.size(); ++j) {
            auto edge_id = node->_edges_id.at(j);
            auto& edge = _edges.at(edge_id);

            auto& n1 = node, n2 = node;
            if (edge->_nodes_id.size() > 1) {
                auto& n1_ = _nodes.at(edge->_nodes_id.at(0));
                auto& n2_ = _nodes.at(edge->_nodes_id.at(1));
                if (n1_->_id == i) {
                    n2 = n2_;
                } else {
                    n2 = n1_;
                }
            }

            _vec_in_edge_direction_x_tot(node->_id, j) = n1->_vectors_in_edges_directions(j, 0);
            _vec_in_edge_direction_y_tot(node->_id, j) = n1->_vectors_in_edges_directions(j, 1);
            _vec_in_edge_neigh_direction_x_tot(node->_id, j) = n2->_vectors_in_edges_directions_by_id.at(edge_id)(0, 0);
            _vec_in_edge_neigh_direction_y_tot(node->_id, j) = n2->_vectors_in_edges_directions_by_id.at(edge_id)(0, 1);

            _n2_ids_tot[j].push_back(n2->_id);
        }
    }

    _volumes = _volumes_tot;
    _normal_x = _normal_x_tot;
    _normal_y = _normal_y_tot;
    _vec_in_edge_direction_x = _vec_in_edge_direction_x_tot;
    _vec_in_edge_direction_y = _vec_in_edge_direction_y_tot;
    _vec_in_edge_neigh_direction_x = _vec_in_edge_neigh_direction_x_tot;
    _vec_in_edge_neigh_direction_y = _vec_in_edge_neigh_direction_y_tot;
    _n2_ids = _n2_ids_tot;
}

void Mesh2D::init_basic_internals() {
    for (int x_ = 0; x_ < _x; ++x_) {
        for (int y_ = 0; y_ < _y; ++y_) {
            auto i = coord_fo_idx(x_, y_);

            if (i == 0) {
                auto v1 = std::make_shared<Vertex2D>(0, 0, 0);
                auto v2 = std::make_shared<Vertex2D>(_dx, 0, 1);
                auto v3 = std::make_shared<Vertex2D>(_dx, _dy, 2);
                auto v4 = std::make_shared<Vertex2D>(0, _dy, 3);

                auto e1 = std::make_shared<Edge2D>(v1->_id, v2->_id, 0);
                auto e2 = std::make_shared<Edge2D>(v2->_id, v3->_id, 1);
                auto e3 = std::make_shared<Edge2D>(v3->_id, v4->_id, 2);
                auto e4 = std::make_shared<Edge2D>(v4->_id, v1->_id, 3);

                auto n = std::make_shared<Quadrangle2D>(e1->_id, e2->_id, e3->_id, e4->_id, v1->_id, v2->_id, v3->_id, v4->_id, 0);

                e1->_nodes_id.push_back(n->_id);
                e2->_nodes_id.push_back(n->_id);
                e3->_nodes_id.push_back(n->_id);
                e4->_nodes_id.push_back(n->_id);

                _vertexes.push_back(v1);
                _vertexes.push_back(v2);
                _vertexes.push_back(v3);
                _vertexes.push_back(v4);
                _nodes.push_back(n);
                _edges.push_back(e1);
                _edges.push_back(e2);
                _edges.push_back(e3);
                _edges.push_back(e4);

            } else if (y_ == 0) {
                auto v2 = std::make_shared<Vertex2D>((x_ + 1) * _dx, 0, _vertexes.size());
                auto v3 = std::make_shared<Vertex2D>((x_ + 1) * _dx, _dy, _vertexes.size() + 1);

                auto node_left_id = coord_fo_idx(x_ - 1, y_);
                auto& v1 = _vertexes.at(_nodes.at(node_left_id)->_vertexes_id.at(1));
                auto& v4 = _vertexes.at(_nodes.at(node_left_id)->_vertexes_id.at(2));

                auto e1 = std::make_shared<Edge2D>(v1->_id, v2->_id, _edges.size());
                auto e2 = std::make_shared<Edge2D>(v2->_id, v3->_id, _edges.size() + 1);
                auto e3 = std::make_shared<Edge2D>(v3->_id, v4->_id, _edges.size() + 2);

                auto& e4 = _edges.at(_nodes.at(node_left_id)->_edges_id.at(1));
                auto n = std::make_shared<Quadrangle2D>(e1->_id, e2->_id, e3->_id, e4->_id, v1->_id, v2->_id, v3->_id, v4->_id, _nodes.size());

                if (std::find(e1->_nodes_id.begin(), e1->_nodes_id.end(), n->_id) == e1->_nodes_id.end()) {
                    e1->_nodes_id.push_back(n->_id);
                }
                if (std::find(e2->_nodes_id.begin(), e2->_nodes_id.end(), n->_id) == e2->_nodes_id.end()) {
                    e2->_nodes_id.push_back(n->_id);
                }
                if (std::find(e3->_nodes_id.begin(), e3->_nodes_id.end(), n->_id) == e3->_nodes_id.end()) {
                    e3->_nodes_id.push_back(n->_id);
                }
                if (std::find(e4->_nodes_id.begin(), e4->_nodes_id.end(), n->_id) == e4->_nodes_id.end()) {
                    e4->_nodes_id.push_back(n->_id);
                }

                _vertexes.push_back(v2);
                _vertexes.push_back(v3);
                _nodes.push_back(n);
                _edges.push_back(e1);
                _edges.push_back(e2);
                _edges.push_back(e3);

            } else if (x_ == 0) {
                auto v3 = std::make_shared<Vertex2D>(_dx, (y_ + 1) * _dy, _vertexes.size());
                auto v4 = std::make_shared<Vertex2D>(0, (y_ + 1) * _dy, _vertexes.size() + 1);

                auto node_bottom_id = coord_fo_idx(x_, y_ - 1);
                auto& v1 = _vertexes.at(_nodes.at(node_bottom_id)->_vertexes_id.at(3));
                auto& v2 = _vertexes.at(_nodes.at(node_bottom_id)->_vertexes_id.at(2));

                auto e2 = std::make_shared<Edge2D>(v2->_id, v3->_id, _edges.size());
                auto e3 = std::make_shared<Edge2D>(v3->_id, v4->_id, _edges.size() + 1);
                auto e4 = std::make_shared<Edge2D>(v4->_id, v1->_id, _edges.size() + 2);

                auto& e1 = _edges.at(_nodes.at(node_bottom_id)->_edges_id.at(2));
                auto n = std::make_shared<Quadrangle2D>(e1->_id, e2->_id, e3->_id, e4->_id, v1->_id, v2->_id, v3->_id, v4->_id, _nodes.size());

                if (std::find(e1->_nodes_id.begin(), e1->_nodes_id.end(), n->_id) == e1->_nodes_id.end()) {
                    e1->_nodes_id.push_back(n->_id);
                }
                if (std::find(e2->_nodes_id.begin(), e2->_nodes_id.end(), n->_id) == e2->_nodes_id.end()) {
                    e2->_nodes_id.push_back(n->_id);
                }
                if (std::find(e3->_nodes_id.begin(), e3->_nodes_id.end(), n->_id) == e3->_nodes_id.end()) {
                    e3->_nodes_id.push_back(n->_id);
                }
                if (std::find(e4->_nodes_id.begin(), e4->_nodes_id.end(), n->_id) == e4->_nodes_id.end()) {
                    e4->_nodes_id.push_back(n->_id);
                }

                _vertexes.push_back(v3);
                _vertexes.push_back(v4);
                _nodes.push_back(n);
                _edges.push_back(e2);
                _edges.push_back(e3);
                _edges.push_back(e4);

            } else {
                auto v3 = std::make_shared<Vertex2D>((x_ + 1)*_dx, (y_ + 1) * _dy, _vertexes.size());

                auto node_left_id = coord_fo_idx(x_ - 1, y_ );
                auto node_bottom_id = coord_fo_idx(x_, y_ - 1);
                auto& v1 = _vertexes.at(_nodes.at(node_left_id)->_vertexes_id.at(1));
                auto& v2 = _vertexes.at(_nodes.at(node_bottom_id)->_vertexes_id.at(2));
                auto& v4 = _vertexes.at(_nodes.at(node_left_id)->_vertexes_id.at(2));

                auto e2 = std::make_shared<Edge2D>(v2->_id, v3->_id, _edges.size());
                auto e3 = std::make_shared<Edge2D>(v3->_id, v4->_id, _edges.size() + 1);

                auto& e1 = _edges.at(_nodes.at(node_bottom_id)->_edges_id.at(2));
                auto& e4 = _edges.at(_nodes.at(node_left_id)->_edges_id.at(1));
                auto n = std::make_shared<Quadrangle2D>(e1->_id, e2->_id, e3->_id, e4->_id, v1->_id, v2->_id, v3->_id, v4->_id, _nodes.size());

                if (std::find(e1->_nodes_id.begin(), e1->_nodes_id.end(), n->_id) == e1->_nodes_id.end()) {
                    e1->_nodes_id.push_back(n->_id);
                }
                if (std::find(e2->_nodes_id.begin(), e2->_nodes_id.end(), n->_id) == e2->_nodes_id.end()) {
                    e2->_nodes_id.push_back(n->_id);
                }
                if (std::find(e3->_nodes_id.begin(), e3->_nodes_id.end(), n->_id) == e3->_nodes_id.end()) {
                    e3->_nodes_id.push_back(n->_id);
                }
                if (std::find(e4->_nodes_id.begin(), e4->_nodes_id.end(), n->_id) == e4->_nodes_id.end()) {
                    e4->_nodes_id.push_back(n->_id);
                }

                _vertexes.push_back(v3);
                _nodes.push_back(n);
                _edges.push_back(e2);
                _edges.push_back(e3);
            }


        }
    }
}

size_t Mesh2D::coord_fo_idx(size_t x, size_t y) const {
    return x * _x + y;
}
