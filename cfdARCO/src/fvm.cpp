#include "fvm.hpp"

#include <utility>
#include <indicators/progress_bar.hpp>

Variable::Variable(Mesh2D *mesh_, Eigen::VectorXd &initial_, BoundaryFN boundary_conditions_, std::string name_) :
        mesh{mesh_}, current{initial_}, boundary_conditions{std::move(boundary_conditions_)}, name{std::move(name_)} {
    num_nodes = mesh->_num_nodes;
}

Variable::Variable(const Variable* left_operand_, const Variable* right_operand_,
                   std::function<Eigen::MatrixXd(Eigen::MatrixXd &, Eigen::MatrixXd &)> op_, std::string &name_) :
                    op{op_}, name{name_}  {
    num_nodes = 0;
    if (left_operand_->mesh != nullptr) {
        mesh = left_operand_->mesh;
    } else if (right_operand_->mesh != nullptr) {
        mesh = right_operand_->mesh;
    }
    is_subvariable = true;
    left_operand = const_cast<Variable*>(left_operand_);
    right_operand = const_cast<Variable*>(right_operand_);
}

Variable::Variable(Mesh2D *mesh_, double value) : mesh{mesh_} {
    current = Eigen::VectorXd {mesh->_num_nodes};
    current.setConstant(value);
    name = std::to_string(value);
    is_constvar = true;
}

Variable::Variable() {
    current = {};
    boundary_conditions = {};
    name = "uninitialized";
    num_nodes = 0;
}

Variable::Variable(Eigen::VectorXd& curr_) {
    current = curr_;
    name = "arr";
    is_constvar = true;
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

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> Variable::get_interface_vars_first_order() {
    if (is_subvariable) {
        auto [left_eval1, left_eval2, left_eval3] = left_operand->get_interface_vars_first_order();
        auto [right_eval1, right_eval2, right_eval3] = right_operand->get_interface_vars_first_order();

        return {op(left_eval1, right_eval1), op(left_eval2, right_eval2), op(left_eval3, right_eval3)};
    }


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

Eigen::MatrixXd Variable::evaluate() {
    if (!is_subvariable) {
        return current;
    }

    auto val_l = left_operand->evaluate();
    auto val_r = right_operand->evaluate();
    return op(val_l, val_r);
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

Variable Variable::operator+(const Variable &obj_r) const {
    std::string name_ = "(" + this->name + "+" + obj_r.name + ")";

    const Variable* l_p;
    const Variable* r_p;
    if (this->is_subvariable || this->is_constvar) {
        l_p = new Variable{*this};
    } else {
        l_p = this;
    }
    if (obj_r.is_subvariable || obj_r.is_constvar) {
        r_p = new Variable{obj_r};
    } else {
        r_p = &obj_r;
    }

    return {l_p, r_p, [](Eigen::MatrixXd& lft, Eigen::MatrixXd& rht){return lft + rht;}, name_};
}

Variable Variable::operator-(const Variable &obj_r) const {
    std::string name_ = "(" + this->name + "-" + obj_r.name + ")";

    const Variable* l_p;
    const Variable* r_p;
    if (this->is_subvariable || this->is_constvar) {
        l_p = new Variable{*this};
    } else {
        l_p = this;
    }
    if (obj_r.is_subvariable || obj_r.is_constvar) {
        r_p = new Variable{obj_r};
    } else {
        r_p = &obj_r;
    }

    return {l_p, r_p, [](Eigen::MatrixXd& lft, Eigen::MatrixXd& rht){return lft - rht;}, name_};
}

Variable Variable::operator*(const Variable &obj_r) const {
    std::string name_ = "(" + this->name + "*" + obj_r.name + ")";

    const Variable* l_p;
    const Variable* r_p;
    if (this->is_subvariable || this->is_constvar) {
        l_p = new Variable{*this};
    } else {
        l_p = this;
    }
    if (obj_r.is_subvariable || obj_r.is_constvar) {
        r_p = new Variable{obj_r};
    } else {
        r_p = &obj_r;
    }

    return {l_p, r_p, [](Eigen::MatrixXd& lft, Eigen::MatrixXd& rht){return lft.array() * rht.array();}, name_};
}

Variable Variable::operator/(const Variable &obj_r) const {
    std::string name_ = "(" + this->name + "/" + obj_r.name + ")";

    const Variable* l_p;
    const Variable* r_p;
    if (this->is_subvariable || this->is_constvar) {
        l_p = new Variable{*this};
    } else {
        l_p = this;
    }
    if (obj_r.is_subvariable || obj_r.is_constvar) {
        r_p = new Variable{obj_r};
    } else {
        r_p = &obj_r;
    }

    return {l_p, r_p, [](Eigen::MatrixXd& lft, Eigen::MatrixXd& rht){return lft.cwiseQuotient(rht);}, name_};
}

Variable Variable::operator-() const {
    std::string name_ = "-(" + this->name + ")";

    const Variable* l_p;
    if (this->is_subvariable || this->is_constvar) {
        l_p = new Variable{*this};
    } else {
        l_p = this;
    }

    return {l_p, l_p, [](Eigen::MatrixXd& lft, Eigen::MatrixXd& rht){return -lft;}, name_};
}


Variable operator+(const double obj_l, const Variable & obj_r) {
    auto val_l = Variable{obj_r.mesh, obj_l};
    return val_l + obj_r;
}

Variable operator-(const double obj_l, const Variable & obj_r) {
    auto val_l = Variable{obj_r.mesh, obj_l};
    return val_l - obj_r;
}

Variable operator*(const double obj_l, const Variable & obj_r) {
    auto val_l = Variable{obj_r.mesh, obj_l};
    return val_l * obj_r;
}

Variable operator/(const double obj_l, const Variable & obj_r) {
    auto val_l = Variable{obj_r.mesh, obj_l};
    return val_l / obj_r;
}

Variable operator+(const Variable & obj_l, const double obj_r) {
    auto val_r = Variable{obj_l.mesh, obj_r};
    return obj_l + val_r;
}

Variable operator-(const Variable & obj_l, const double obj_r) {
    auto val_r = Variable{obj_l.mesh, obj_r};
    return obj_l - val_r;
}

Variable operator*(const Variable & obj_l, const double obj_r) {
    auto val_r = Variable{obj_l.mesh, obj_r};
    return obj_l * val_r;
}

Variable operator/(const Variable & obj_l, const double obj_r) {
    auto val_r = Variable{obj_l.mesh, obj_r};
    return obj_l / val_r;
}

_GradEstimated::_GradEstimated(Variable *var_, bool clc_x_, bool clc_y_) : var{var_}, clc_x{clc_x_}, clc_y{clc_y_} {
    mesh = var->mesh;
    name = "GradEstimated(" + var_->name + ")";
    is_constvar = true;
}

Eigen::MatrixXd _GradEstimated::evaluate() {
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
    auto gamma = 5. / 3.;
//    auto l = space_vars.at(5);

    auto dl = 0.01;
    auto denom = dl * (((gamma * p->current.array()).cwiseQuotient(rho->current.array())).cwiseSqrt() + (u->current.array() * u->current.array() + v->current.array() * v->current.array()).cwiseSqrt()).cwiseInverse();

    auto dt = CFL * denom.minCoeff();

    return dt;
}

DT::DT(Mesh2D* mesh_, std::function<double(double, std::vector<Variable *> &)> update_fn_, double CFL_, std::vector<Variable *> &space_vars_) : update_fn{update_fn_}, CFL{CFL_}, space_vars{space_vars_} {
    name = "dt";
    mesh = mesh_;
    _dt = 0;
}

void DT::update() {
    _dt = update_fn(CFL, space_vars);
}

Eigen::MatrixXd DT::evaluate() {
    auto ret = Eigen::VectorXd{mesh->_num_nodes};
    ret.setConstant(_dt);
    return ret;
}

_DT::_DT(Variable *var_) : var{var_} {}

Eigen::VectorXd _DT::extract(Eigen::VectorXd& left_part, double dt) {
    return dt * left_part + var->current;
}

void _DT::solve(Variable& equation, DT& dt) {
    EqSolver::solve_dt(&equation, this, var, &dt);
}

_Grad::_Grad(Variable *var_, bool clc_x_, bool clc_y_) : var{var_}, clc_x{clc_x_}, clc_y{clc_y_} {}

Eigen::MatrixXd _Grad::evaluate() {
    auto current_interface_gradients = var->get_interface_vars_first_order();
    auto& current_interface_gradients_star = std::get<0>(current_interface_gradients);

    auto res_x = (current_interface_gradients_star * var->mesh->_normal_x).rowwise().sum();
    auto res_y = (current_interface_gradients_star * var->mesh->_normal_y).rowwise().sum();

    if (clc_x && clc_y) {
        return res_x + res_y;
    }
    if (clc_x) {
        return res_x;
    }
    if (clc_y) {
        return res_y;
    }
}

_Stab::_Stab(Variable *var_, bool clc_x_, bool clc_y_) : var{var_}, clc_x{clc_x_}, clc_y{clc_y_} {}

Eigen::MatrixXd _Stab::evaluate() {
    auto current_interface_gradients = var->get_interface_vars_first_order();
    auto& current_interface_gradients_star = (std::get<2>(current_interface_gradients) - std::get<1>(current_interface_gradients)) / 2;

    auto res_x = (current_interface_gradients_star * var->mesh->_normal_x).rowwise().sum();
    auto res_y = (current_interface_gradients_star * var->mesh->_normal_y).rowwise().sum();

    if (clc_x && clc_y) {
        return res_x + res_y;
    }
    if (clc_x) {
        return res_x;
    }
    if (clc_y) {
        return res_y;
    }
}

void EqSolver::solve_dt(Variable *equation, Variable *time_var, Variable *set_var, DT *dt) {
    Eigen::VectorXd current = equation->evaluate();
    auto extracted = time_var->extract(current, dt->_dt);
    set_var->set_current(extracted);
}

Equation::Equation(size_t timesteps_) : timesteps{timesteps_} {}

void Equation::evaluate(std::vector<Variable> &all_vars,
                        std::vector<std::tuple<Variable, char, Variable>> &equation_system, DT& dt) {
    double t_val = 0;
    indicators::ProgressBar bar{
            indicators::option::BarWidth{50},
            indicators::option::Start{"["},
            indicators::option::Fill{"="},
            indicators::option::Lead{">"},
            indicators::option::Remainder{" "},
            indicators::option::End{"]"},
            indicators::option::ForegroundColor{indicators::Color::green},
            indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}
    };

    for (int t = 0; t < timesteps; ++t) {
        dt.update();

        for (auto var : all_vars) {
            var.set_bound();
        }

        t_val += dt._dt;

        for (auto& equation : equation_system) {
            auto& left_part = std::get<0>(equation);
            auto& right_part = std::get<2>(equation);
            left_part.solve(right_part, dt);
        }

        for (auto var : all_vars) {
            var.set_bound();
            var.add_history();
        }

        double progress = (static_cast<double>(t) / static_cast<double>(timesteps)) * 100;
        bar.set_progress(static_cast<size_t>(progress));
    }
}
