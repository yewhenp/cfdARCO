#include "fvm.hpp"

#include <utility>

Variable::Variable(Mesh2D *mesh_, Eigen::VectorXd &initial_, BoundaryFN boundary_conditions_, std::string name_) :
        mesh{mesh_}, current{initial_}, boundary_conditions{std::move(boundary_conditions_)}, name{std::move(name_)} {
    num_nodes = mesh->_num_nodes;
}

Variable::Variable(Variable* left_operand_, Variable* right_operand_,
                   std::function<Eigen::MatrixXd(Eigen::MatrixXd &, Eigen::MatrixXd &)> op_, std::string &name_) :
                   left_operand{left_operand_}, right_operand{right_operand_}, op{op_}, name{name_}  {
    num_nodes = 0;
    is_subvariable = true;
}

Variable::Variable() {
    current = {};
    boundary_conditions = {};
    name = "uninitialized";
    num_nodes = 0;
}


void Variable::set_bound() {
    current = boundary_conditions(mesh, current);
}

void Variable::add_history() {
    history.push_back({current});
}

Eigen::MatrixXd Variable::estimate_grads() {
    auto ret = Eigen::Matrix<double, -1, 2> { num_nodes, 2};

    for (int i = 0; i < num_nodes; ++i) {
        auto& node = mesh->_nodes.at(i);
        auto summ = Eigen::Matrix<double, 1, 2>{};
        for (int j = 0; j < node._edges_id.size(); ++j) {
            auto edge_id = node._edges_id.at(j);
            auto& edge = mesh->_edges.at(edge_id);

            Quadrangle2D& n1 = node, n2 = node;
            if (edge._nodes_id.size() > 1) {
                auto& n1_ = mesh->_nodes.at(edge._nodes_id.at(0));
                auto& n2_ = mesh->_nodes.at(edge._nodes_id.at(1));
                if (n1_._id == i) {
                    n2 = n2_;
                } else {
                    n2 = n1_;
                }
            }

            auto fi = (current(n1._id) + current(n2._id)) / 2;
            summ = summ + (n1._normals.block<1, 2>(j, 0) * fi);
        }
        ret.block<1, 2>(i, 0) += summ;
    }

    return ret;
}

_GradEstimated Variable::dx() {
    return _GradEstimated{this, true, false};
}

_GradEstimated Variable::dy() {
    return _GradEstimated{this,  false, true};
}

std::tuple<Eigen::MatrixX4d, Eigen::MatrixX4d, Eigen::MatrixX4d> Variable::get_interface_vars_first_order() {
    auto grads = estimate_grads();
    auto ret_sum = Eigen::Matrix<double, -1, 4> { num_nodes, 4};
    auto ret_r = Eigen::Matrix<double, -1, 4> { num_nodes, 4};
    auto ret_l = Eigen::Matrix<double, -1, 4> { num_nodes, 4};

    for (int i = 0; i < num_nodes; ++i) {
        auto& node = mesh->_nodes.at(i);

        for (int j = 0; j < node._edges_id.size(); ++j) {
            auto edge_id = node._edges_id.at(j);
            auto& edge = mesh->_edges.at(edge_id);

            Quadrangle2D& n1 = node, n2 = node;
            if (edge._nodes_id.size() > 1) {
                auto& n1_ = mesh->_nodes.at(edge._nodes_id.at(0));
                auto& n2_ = mesh->_nodes.at(edge._nodes_id.at(1));
                if (n1_._id == i) {
                    n2 = n2_;
                } else {
                    n2 = n1_;
                }
            }

            auto n1_to_mid = n1._vectors_in_edges_directions.block<1, 2>(j, 0);
            auto n2_to_mid = n2._vectors_in_edges_directions_by_id.at(edge_id);

            auto fi_n1 =  grads.block<1, 2>(n1._id, 0).dot(n1_to_mid) + current(n1._id);
            auto fi_n2 =  grads.block<1, 2>(n2._id, 0).dot(n2_to_mid) + current(n2._id);

            ret_sum(i, j) = (fi_n1 + fi_n2) / 2;

            if (j == 0) {
                ret_r(i, j) = fi_n2;
                ret_l(i, j) = fi_n1;
            } else if (j == 1) {
                ret_r(i, j) = fi_n1;
                ret_l(i, j) = fi_n2;
            } else if (j == 2) {
                ret_r(i, j) = fi_n1;
                ret_l(i, j) = fi_n2;
            } else {
                ret_r(i, j) = fi_n2;
                ret_l(i, j) = fi_n1;
            }
        }
    }

    return {ret_sum, ret_r, ret_l};
}

Eigen::VectorXd Variable::extract(Eigen::VectorXd &left_part, double dt) {
    return left_part;
}

Eigen::VectorXd Variable::evaluate() {
    return current;
}

void Variable::set_current(Eigen::VectorXd &current_) {
    current = current_;
}

std::vector<Eigen::VectorXd> Variable::get_history() {
    return history;
}

void Variable::solve(Variable &equation, DT &dt) {
    EqSolver::solve_dt(&equation, this, this, &dt);
}

Variable Variable::operator+(Variable &obj_r) {
    std::string name_ = "add";
    return {this, &obj_r, [](Eigen::MatrixXd& lft, Eigen::MatrixXd& rht){return lft + rht;}, name_};
}

Variable Variable::operator-(Variable &obj_r) {
    std::string name_ = "sub";
    return {this, &obj_r, [](Eigen::MatrixXd& lft, Eigen::MatrixXd& rht){return lft - rht;}, name_};
}

Variable Variable::operator*(Variable &obj_r) {
    std::string name_ = "mul";
    return {this, &obj_r, [](Eigen::MatrixXd& lft, Eigen::MatrixXd& rht){return lft * rht;}, name_};
}

Variable Variable::operator/(Variable &obj_r) {
    std::string name_ = "div";
    return {this, &obj_r, [](Eigen::MatrixXd& lft, Eigen::MatrixXd& rht){return lft.cwiseQuotient(rht);}, name_};
}

Variable Variable::operator-() {
    std::string name_ = "neg";
    return {this, this, [](Eigen::MatrixXd& lft, Eigen::MatrixXd& rht){return -lft;}, name_};
}


_GradEstimated::_GradEstimated(Variable *var_, bool clc_x_, bool clc_y_) : var{var_}, clc_x{clc_x_}, clc_y{clc_y_} {}

Eigen::VectorXd _GradEstimated::evaluate() {
    auto grads = var->estimate_grads();
    if (clc_x && clc_y) {
        return grads.col(0) + grads.col(1);
    }
    if (clc_x) {
        return grads.col(0);
    }
    if (clc_y) {
        return grads.col(1);
    }
}

// TODO: think about general interface
double UpdatePolicies::CourantFriedrichsLewy(double CFL, std::vector<Variable *> &space_vars) {
    auto u = space_vars.at(0);
    auto v = space_vars.at(1);
    auto p = space_vars.at(2);
    auto rho = space_vars.at(3);
    auto gamma = space_vars.at(4);
//    auto l = space_vars.at(5);

    auto dl = 0.01;
    auto denom = dl * (((gamma->current * p->current).cwiseQuotient(rho->current)).cwiseSqrt() + (u->current * u->current + v->current * v->current).cwiseSqrt()).cwiseInverse();

    auto dt = CFL * denom.minCoeff();

    return 0;
}

DT::DT(Mesh2D* mesh_, std::function<double(double, std::vector<Variable *> &)> update_fn_, double CFL_, std::vector<Variable *> &space_vars_) : mesh{mesh_}, update_fn{update_fn_}, CFL{CFL_}, space_vars{space_vars_} {
    name = "dt";
    _dt = 0;
}

void DT::update() {
    _dt = update_fn(CFL, space_vars);
}

Eigen::VectorXd DT::evaluate() {
    auto ret = Eigen::VectorXd{mesh->_num_nodes};
    ret.setConstant(_dt);
    return ret;
}
