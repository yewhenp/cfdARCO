// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com


#include "fvm.hpp"
#include "cfdarcho_main.hpp"

#include <utility>
#include <indicators/progress_bar.hpp>

Variable::Variable(Mesh2D* mesh_, Eigen::VectorXd &initial_, BoundaryFN boundary_conditions_, std::string name_) :
        mesh{mesh_}, current{initial_}, boundary_conditions{std::move(boundary_conditions_)}, name{std::move(name_)} {
    num_nodes = mesh->_num_nodes;
    is_basically_created = true;
}

Variable::Variable(const std::shared_ptr<Variable> left_operand_, const std::shared_ptr<Variable> right_operand_,
                   std::function<MatrixX4dRB(MatrixX4dRB &, MatrixX4dRB &)> op_, std::string &name_) :
                    op{op_}, name{name_}  {
    num_nodes = 0;
    if (left_operand_->mesh != nullptr) {
        mesh = left_operand_->mesh;
        num_nodes = mesh->_num_nodes;
    } else if (right_operand_->mesh != nullptr) {
        mesh = right_operand_->mesh;
        num_nodes = mesh->_num_nodes;
    }
    is_subvariable = true;
    is_basically_created = false;
    left_operand = left_operand_;
    right_operand = right_operand_;
}

Variable::Variable(Mesh2D* mesh_, double value) : mesh{mesh_} {
    current = Eigen::VectorXd {mesh->_num_nodes};
    num_nodes = mesh->_num_nodes;
    current.setConstant(value);
    name = std::to_string(value);
    is_constvar = true;
    is_basically_created = false;
}

Variable::Variable() {
    current = {};
    boundary_conditions = {};
    name = "uninitialized";
    num_nodes = 0;
    is_basically_created = false;
}

Variable::Variable(Eigen::VectorXd& curr_) {
    current = curr_;
    num_nodes = curr_.rows();
    name = "arr";
    is_constvar = true;
    is_basically_created = false;
}

Variable::Variable(Variable &copy_var) {
    name = copy_var.name;
    mesh = copy_var.mesh;
    current = copy_var.current;
    boundary_conditions = copy_var.boundary_conditions;
    history = copy_var.history;
    num_nodes = copy_var.num_nodes;
    is_subvariable = copy_var.is_subvariable;
    is_constvar = copy_var.is_constvar;
    op = copy_var.op;
    is_basically_created = false;

    if (copy_var.left_operand) {
        left_operand = std::shared_ptr<Variable> {copy_var.left_operand->clone()};
    }
    if (copy_var.right_operand) {
        right_operand = std::shared_ptr<Variable> {copy_var.right_operand->clone()};
    }
}

Variable::Variable(const Variable &copy_var) {
    name = copy_var.name;
    mesh = copy_var.mesh;
    current = copy_var.current;
    boundary_conditions = copy_var.boundary_conditions;
    history = copy_var.history;
    num_nodes = copy_var.num_nodes;
    is_subvariable = copy_var.is_subvariable;
    is_constvar = copy_var.is_constvar;
    op = copy_var.op;
    is_basically_created = false;

    if (copy_var.left_operand) {
        left_operand = std::shared_ptr<Variable> {copy_var.left_operand->clone()};
    }
    if (copy_var.right_operand) {
        right_operand = std::shared_ptr<Variable> {copy_var.right_operand->clone()};
    }
}

std::shared_ptr<Variable> Variable::clone() const {
    if (is_basically_created) {
        return std::shared_ptr<Variable>{const_cast<Variable*>(this), [](Variable *) {}};
    }
    auto* new_obj = new Variable(*this);
    return std::shared_ptr<Variable>{new_obj};
}

std::shared_ptr<Variable> _GradEstimated::clone() const {
    auto* new_obj = new _GradEstimated(*this);
    return std::shared_ptr<Variable>{new_obj};
}

std::shared_ptr<Variable> DT::clone() const {
    return std::shared_ptr<Variable>{const_cast<DT*>(this), [](Variable *) {}};
}

std::shared_ptr<Variable> _DT::clone() const {
    auto* new_obj = new _DT(*this);
    return std::shared_ptr<Variable>{new_obj};
}

std::shared_ptr<Variable> _Grad::clone() const {
    auto* new_obj = new _Grad(*this);
    return std::shared_ptr<Variable>{new_obj};
}

std::shared_ptr<Variable> _Grad2::clone() const {
    auto* new_obj = new _Grad2(*this);
    return std::shared_ptr<Variable>{new_obj};
}

std::shared_ptr<Variable> _Stab::clone() const  {
    auto* new_obj = new _Stab(*this);
    return std::shared_ptr<Variable>{new_obj};
}


void Variable::set_bound() {
    current = boundary_conditions(mesh, current);
}

void Variable::add_history() {
    history.push_back({current});
}

MatrixX4dRB Variable::estimate_grads() {
    if (estimate_grid_cache_valid) {
        return estimate_grid_cache;
    }

    estimate_grid_cache = Eigen::Matrix<double, -1, 2> { num_nodes, 2};
    estimate_grid_cache.setConstant(0);
    auto pre_ret = Eigen::Matrix<double, -1, 4> { num_nodes, 4};

    auto send_name = name + "current";
    current_redist = CFDArcoGlobalInit::get_redistributed(current, send_name);

    for (int j = 0; j < 4; ++j) {
//        auto current_n2 = current(mesh->_n2_ids[j]);
        auto current_n2 = current_redist[j];
        pre_ret.col(j) = (current + current_n2) / 2;
    }
    auto pre_ret_x = pre_ret.cwiseProduct(mesh->_normal_x);
    auto pre_ret_y = pre_ret.cwiseProduct(mesh->_normal_y);

    auto grad_x = pre_ret_x.rowwise().sum();
    auto grad_y = pre_ret_y.rowwise().sum();

    auto grad_xd = grad_x.cwiseQuotient(mesh->_volumes);
    auto grad_yd = grad_y.cwiseQuotient(mesh->_volumes);

    estimate_grid_cache.col(0) = grad_xd;
    estimate_grid_cache.col(1) = grad_yd;
    estimate_grid_cache_valid = true;

    send_name = name + "grad";
    grad_redist = CFDArcoGlobalInit::get_redistributed(estimate_grid_cache, send_name);

    return estimate_grid_cache;
}

_GradEstimated Variable::dx() {
    return _GradEstimated{this, true, false};
}

_GradEstimated Variable::dy() {
    return _GradEstimated{this,  false, true};
}

std::tuple<MatrixX4dRB, MatrixX4dRB, MatrixX4dRB> Variable::get_interface_vars_second_order() {
    if (is_subvariable) {
        auto [left_eval1, left_eval2, left_eval3] = left_operand->get_interface_vars_second_order();
        auto [right_eval1, right_eval2, right_eval3] = right_operand->get_interface_vars_second_order();

        return {op(left_eval1, right_eval1), op(left_eval2, right_eval2), op(left_eval3, right_eval3)};
    }

    auto grads = estimate_grads();
    auto grads_self_x = Eigen::Matrix<double, -1, 4> { num_nodes, 4};
    auto grads_self_y = Eigen::Matrix<double, -1, 4> { num_nodes, 4};
    auto grads_neigh_x = Eigen::Matrix<double, -1, 4> { num_nodes, 4};
    auto grads_neigh_y = Eigen::Matrix<double, -1, 4> { num_nodes, 4};
    auto cur_self = Eigen::Matrix<double, -1, 4> { num_nodes, 4};
    auto cur_neigh = Eigen::Matrix<double, -1, 4> { num_nodes, 4};

    auto ret_r = Eigen::Matrix<double, -1, 4> { num_nodes, 4};
    auto ret_l = Eigen::Matrix<double, -1, 4> { num_nodes, 4};

    for (int j = 0; j < 4; ++j) {
        grads_self_x.col(j) = grads.col(0);
        grads_self_y.col(j) = grads.col(1);
        grads_neigh_x.col(j) = grad_redist[j].col(0);
        grads_neigh_y.col(j) = grad_redist[j].col(1);
        cur_self.col(j) = current;
        cur_neigh.col(j) = current_redist[j];
    }

    auto grads_self_xd = grads_self_x.cwiseProduct(mesh->_vec_in_edge_direction_x);
    auto grads_self_yd = grads_self_y.cwiseProduct(mesh->_vec_in_edge_direction_y);
    auto grads_neigh_xd = grads_neigh_x.cwiseProduct(mesh->_vec_in_edge_neigh_direction_x);
    auto grads_neigh_yd = grads_neigh_y.cwiseProduct(mesh->_vec_in_edge_neigh_direction_y);

    auto val_self = (grads_self_xd + grads_self_yd + cur_self).eval();
    auto val_neigh = (grads_neigh_xd + grads_neigh_yd + cur_neigh).eval();

    auto ret_sum = (val_self + val_neigh) / 2;

    ret_r.col(0) = val_neigh.col(0);
    ret_r.col(1) = val_self.col(1);
    ret_r.col(2) = val_self.col(2);
    ret_r.col(3) = val_neigh.col(3);

    ret_l.col(0) = val_self.col(0);
    ret_l.col(1) = val_neigh.col(1);
    ret_l.col(2) = val_neigh.col(2);
    ret_l.col(3) = val_self.col(3);

    return {ret_sum, ret_r, ret_l};
}

std::tuple<MatrixX4dRB, MatrixX4dRB, MatrixX4dRB> Variable::get_interface_vars_first_order() {
    if (is_subvariable) {
        auto [left_eval1, left_eval2, left_eval3] = left_operand->get_interface_vars_first_order();
        auto [right_eval1, right_eval2, right_eval3] = right_operand->get_interface_vars_first_order();

        return {op(left_eval1, right_eval1), op(left_eval2, right_eval2), op(left_eval3, right_eval3)};
    }

    auto grads = estimate_grads();
    auto grads_self_x = Eigen::Matrix<double, -1, 4> { num_nodes, 4};
    auto grads_self_y = Eigen::Matrix<double, -1, 4> { num_nodes, 4};
    auto grads_neigh_x = Eigen::Matrix<double, -1, 4> { num_nodes, 4};
    auto grads_neigh_y = Eigen::Matrix<double, -1, 4> { num_nodes, 4};
    auto cur_self = Eigen::Matrix<double, -1, 4> { num_nodes, 4};
    auto cur_neigh = Eigen::Matrix<double, -1, 4> { num_nodes, 4};

    auto ret_r = Eigen::Matrix<double, -1, 4> { num_nodes, 4};
    auto ret_l = Eigen::Matrix<double, -1, 4> { num_nodes, 4};

    for (int j = 0; j < 4; ++j) {
//        auto grads_n2 = grads( mesh->_n2_ids[j], Eigen::all);
//        auto current_n2 = current( mesh->_n2_ids[j]);
//      assume that current_redist and grad_redist are set

        grads_self_x.col(j) = grads.col(0);
        grads_self_y.col(j) = grads.col(1);
//        grads_neigh_x.col(j) = grads_n2.col(0);
//        grads_neigh_y.col(j) = grads_n2.col(1);
        grads_neigh_x.col(j) = grad_redist[j].col(0);
        grads_neigh_y.col(j) = grad_redist[j].col(1);
        cur_self.col(j) = current;
//        cur_neigh.col(j) = current_n2;

        cur_neigh.col(j) = current_redist[j];
    }

    auto grads_self_xd = grads_self_x.cwiseProduct(mesh->_vec_in_edge_direction_x);
    auto grads_self_yd = grads_self_y.cwiseProduct(mesh->_vec_in_edge_direction_y);
    auto grads_neigh_xd = grads_neigh_x.cwiseProduct(mesh->_vec_in_edge_neigh_direction_x);
    auto grads_neigh_yd = grads_neigh_y.cwiseProduct(mesh->_vec_in_edge_neigh_direction_y);

    auto val_self = (grads_self_xd + grads_self_yd + cur_self).eval();
    auto val_neigh = (grads_neigh_xd + grads_neigh_yd + cur_neigh).eval();

    auto ret_sum = (val_self + val_neigh) / 2;

    ret_r.col(0) = val_neigh.col(0);
    ret_r.col(1) = val_self.col(1);
    ret_r.col(2) = val_self.col(2);
    ret_r.col(3) = val_neigh.col(3);

    ret_l.col(0) = val_self.col(0);
    ret_l.col(1) = val_neigh.col(1);
    ret_l.col(2) = val_neigh.col(2);
    ret_l.col(3) = val_self.col(3);

    return {ret_sum, ret_r, ret_l};
}


Eigen::VectorXd Variable::extract(Eigen::VectorXd &left_part, double dt) {
    return left_part;
}

MatrixX4dRB Variable::evaluate() {
    if (!is_subvariable) {
        return current;
    }

    auto val_l = left_operand->evaluate();
    auto val_r = right_operand->evaluate();
    return op(val_l, val_r);
}

void Variable::set_current(Eigen::VectorXd &current_) {
    current = current_;
    estimate_grid_cache_valid = false;
}

std::vector<Eigen::VectorXd> Variable::get_history() {
    return history;
}

void Variable::solve(Variable* equation, DT* dt) {
    EqSolver::solve_dt(equation, this, this, dt);
}

std::tuple<std::shared_ptr<Variable>, std::shared_ptr<Variable>> get_that_vars(const Variable* obj_l, const Variable &obj_r) {
    std::shared_ptr<Variable> l_p;
    std::shared_ptr<Variable> r_p;
    if (obj_l->is_subvariable || obj_l->is_constvar) {
        l_p = std::shared_ptr<Variable> {obj_l->clone()};
    } else {
        l_p = std::shared_ptr<Variable>(const_cast<Variable*>(obj_l), [](Variable *) {});
    }
    if (obj_r.is_subvariable || obj_r.is_constvar) {
        r_p = std::shared_ptr<Variable> {obj_r.clone()};
    } else {
        r_p = std::shared_ptr<Variable>(const_cast<Variable*>(&obj_r), [](Variable *) {});
    }

    return {l_p, r_p};
}

Variable Variable::operator+(const Variable &obj_r) const {
    std::string name_ = "(" + this->name + "+" + obj_r.name + ")";
    auto [l_p, r_p] = get_that_vars(this, obj_r);
    return {l_p, r_p, [](MatrixX4dRB& lft, MatrixX4dRB& rht){return lft + rht;}, name_};
}

Variable Variable::operator-(const Variable &obj_r) const {
    std::string name_ = "(" + this->name + "-" + obj_r.name + ")";
    auto [l_p, r_p] = get_that_vars(this, obj_r);
    return {l_p, r_p, [](MatrixX4dRB& lft, MatrixX4dRB& rht){return lft - rht;}, name_};
}

Variable Variable::operator*(const Variable &obj_r) const {
    std::string name_ = "(" + this->name + "*" + obj_r.name + ")";
    auto [l_p, r_p] = get_that_vars(this, obj_r);
    return {l_p, r_p, [](MatrixX4dRB& lft, MatrixX4dRB& rht){return lft.cwiseProduct(rht);}, name_};
}

Variable Variable::operator/(const Variable &obj_r) const {
    std::string name_ = "(" + this->name + "/" + obj_r.name + ")";
    auto [l_p, r_p] = get_that_vars(this, obj_r);
    return {l_p, r_p, [](MatrixX4dRB& lft, MatrixX4dRB& rht){return lft.cwiseQuotient(rht);}, name_};
}

Variable Variable::operator-() const {
    std::string name_ = "-(" + this->name + ")";
    auto [l_p, r_p] = get_that_vars(this, *this);
    return {l_p, l_p, [](MatrixX4dRB& lft, MatrixX4dRB& rht){return -lft;}, name_};
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

MatrixX4dRB _GradEstimated::evaluate() {
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
    return MatrixX4dRB{};
}

// TODO: think about general interface
double UpdatePolicies::CourantFriedrichsLewy(double CFL, std::vector<Variable *> &space_vars, Mesh2D* mesh) {
    auto u = space_vars.at(0);
    auto v = space_vars.at(1);
    auto p = space_vars.at(2);
    auto rho = space_vars.at(3);
    auto gamma = 5. / 3.;
    double dl = std::min(mesh->_dx, mesh->_dy);
    auto denom = dl * (((gamma * p->current.array()).cwiseQuotient(rho->current.array())).cwiseSqrt() + (u->current.array() * u->current.array() + v->current.array() * v->current.array()).cwiseSqrt()).cwiseInverse();

    auto dt = CFL * denom.minCoeff();

    return dt;
}

DT::DT(Mesh2D* mesh_, std::function<double(double, std::vector<Variable *> &, Mesh2D* mesh)> update_fn_, double CFL_, std::vector<Variable *> &space_vars_) : update_fn{update_fn_}, CFL{CFL_}, space_vars{space_vars_} {
    name = "dt";
    mesh = mesh_;
    _dt = 0;
}

void DT::update() {
    _dt = update_fn(CFL, space_vars, mesh);
}

MatrixX4dRB DT::evaluate() {
    auto ret = Eigen::VectorXd{mesh->_num_nodes};
    ret.setConstant(_dt);
    return ret;
}

_DT::_DT(Variable *var_, int) {
    var = std::shared_ptr<Variable> {var_, [](Variable *) {}};
}

Eigen::VectorXd _DT::extract(Eigen::VectorXd& left_part, double dt) {
    return dt * left_part + var->current;
}

void _DT::solve(Variable* equation, DT* dt) {
    EqSolver::solve_dt(equation, this, var.get(), dt);
}

_Grad::_Grad(Variable *var_, bool clc_x_, bool clc_y_) : clc_x{clc_x_}, clc_y{clc_y_} {
    var = std::shared_ptr<Variable>{var_->clone()};
    mesh = var_->mesh;
}

MatrixX4dRB _Grad::evaluate() {
    auto current_interface_gradients = var->get_interface_vars_first_order();
    auto& current_interface_gradients_star = std::get<0>(current_interface_gradients);

    auto res_x = (current_interface_gradients_star.cwiseProduct(var->mesh->_normal_x)).rowwise().sum();
    auto res_y = (current_interface_gradients_star.cwiseProduct(var->mesh->_normal_y)).rowwise().sum();

    if (clc_x && clc_y) {
        return res_x + res_y;
    }
    if (clc_x) {
        return res_x;
    }
    if (clc_y) {
        return res_y;
    }
    return MatrixX4dRB{};
}

_Grad2::_Grad2(Variable *var_, bool clc_x_, bool clc_y_) : clc_x{clc_x_}, clc_y{clc_y_} {
    var = std::shared_ptr<Variable>{var_->clone()};
    mesh = var_->mesh;
}

MatrixX4dRB _Grad2::evaluate() {
    auto current_interface_gradients = var->get_interface_vars_first_order();
    auto& current_interface_gradients_star = std::get<0>(current_interface_gradients);

    Eigen::VectorXd res_x = (current_interface_gradients_star.cwiseProduct(var->mesh->_normal_x)).rowwise().sum();
    Eigen::VectorXd res_y = (current_interface_gradients_star.cwiseProduct(var->mesh->_normal_y)).rowwise().sum();

    if (clc_x) {
        return res_x;
    }
    if (clc_y) {
        return res_y;
    }
    return MatrixX4dRB{};

//    auto current_interface_gradients = var->get_interface_vars_first_order();
//    auto& current_interface_gradients_star = std::get<0>(current_interface_gradients);
//
//    Eigen::VectorXd res_x = (current_interface_gradients_star.cwiseProduct(var->mesh->_normal_x)).rowwise().sum();
//    Eigen::VectorXd res_y = (current_interface_gradients_star.cwiseProduct(var->mesh->_normal_y)).rowwise().sum();
//
//    if (clc_x) {
//        auto varr = Variable(mesh, res_x, [](Mesh2D* mesh, Eigen::VectorXd& arr){ return arr; }, "tmp");
//        return d1dx(varr).evaluate();
//    }
//    if (clc_y) {
//        auto varr = Variable(mesh, res_y, [](Mesh2D* mesh, Eigen::VectorXd& arr){ return arr; }, "tmp");
//        return d1dy(varr).evaluate();
//    }
//    return MatrixX4dRB{};

//    auto current_interface_gradients = var->get_interface_vars_first_order();
//    auto& current_interface_gradients_star = std::get<0>(current_interface_gradients);
//
//    Eigen::VectorXd res_x = (current_interface_gradients_star.cwiseProduct(var->mesh->_normal_x)).rowwise().sum();
//    Eigen::VectorXd res_y = (current_interface_gradients_star.cwiseProduct(var->mesh->_normal_y)).rowwise().sum();
//
//    if (clc_x) {
//        auto varr = Variable(mesh, res_x, [](Mesh2D* mesh, Eigen::VectorXd& arr){ return arr; }, "tmp");
//        return varr.dx().evaluate();
//    }
//    if (clc_y) {
//        auto varr = Variable(mesh, res_y, [](Mesh2D* mesh, Eigen::VectorXd& arr){ return arr; }, "tmp");
//        return varr.dy().evaluate();
//    }
//    return MatrixX4dRB{};

//if (clc_x) {
//        return var->dx().evaluate();
//    }
//    if (clc_y) {
//        return var->dy().evaluate();
//    }
//    return MatrixX4dRB{};
}

_Stab::_Stab(Variable *var_, bool clc_x_, bool clc_y_) : clc_x{clc_x_}, clc_y{clc_y_} {
    var = std::shared_ptr<Variable>{var_->clone()};
    mesh = var_->mesh;
}

MatrixX4dRB _Stab::evaluate() {
    auto current_interface_gradients = var->get_interface_vars_first_order();
    auto current_interface_gradients_star = (std::get<2>(current_interface_gradients) - std::get<1>(current_interface_gradients)) / 2;

    auto res_x = (current_interface_gradients_star.cwiseProduct(var->mesh->_normal_x)).rowwise().sum();
    auto res_y = (current_interface_gradients_star.cwiseProduct(var->mesh->_normal_y)).rowwise().sum();

    if (clc_x && clc_y) {
        return res_x + res_y;
    }
    if (clc_x) {
        return res_x;
    }
    if (clc_y) {
        return res_y;
    }
    return MatrixX4dRB{};
}

void EqSolver::solve_dt(Variable *equation, Variable *time_var, Variable *set_var, DT *dt) {
    Eigen::VectorXd current = equation->evaluate();
    auto extracted = time_var->extract(current, dt->_dt);
    set_var->set_current(extracted);
}

Equation::Equation(size_t timesteps_) : timesteps{timesteps_} {}

void Equation::evaluate(std::vector<Variable*> &all_vars,
                        std::vector<std::tuple<Variable*, char, Variable>> &equation_system, DT* dt, bool visualize) {
    double t_val = 0;
    indicators::ProgressBar bar{
            indicators::option::BarWidth{50},
            indicators::option::Start{"["},
            indicators::option::Fill{"="},
            indicators::option::Lead{">"},
            indicators::option::Remainder{" "},
            indicators::option::End{"]"},
            indicators::option::PostfixText{"0.0 %"},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true},
            indicators::option::ForegroundColor{indicators::Color::green},
            indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}
    };

    for (int t = 0; t < timesteps; ++t) {
        dt->update();

        for (auto var : all_vars) {
            var->set_bound();
        }

        t_val += dt->_dt;

        for (auto& equation : equation_system) {
            auto left_part = std::get<0>(equation);
            auto& right_part = std::get<2>(equation);
            left_part->solve(&right_part, dt);
        }

        for (auto var : all_vars) {
            var->set_bound();
            var->add_history();
        }

        if (visualize && CFDArcoGlobalInit::get_rank() == 0) {
            double progress = (static_cast<double>(t) / static_cast<double>(timesteps)) * 100;
            bar.set_progress(static_cast<size_t>(progress));
            bar.set_option(indicators::option::PostfixText{std::to_string(progress) + " %"});

//            std::cout << t_val << std::endl;
        }
    }
}

MatrixX4dRB to_grid(Mesh2D* mesh, Eigen::VectorXd& values_half) {
    auto values = CFDArcoGlobalInit::recombine(values_half, "to_grid");
    MatrixX4dRB result = {mesh->_x, mesh->_y};
    for (auto& node : mesh->_nodes_tot) {
        double value = values(node->_id);
        size_t x_coord = node->_id / mesh->_x;
        size_t y_coord = node->_id % mesh->_x;
        result(x_coord, y_coord) = value;
    }
    return result;
}
