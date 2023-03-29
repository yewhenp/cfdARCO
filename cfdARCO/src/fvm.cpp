#include "fvm.hpp"

#include <utility>
#include <indicators/progress_bar.hpp>

Variable::Variable(Mesh2D* mesh_, Eigen::VectorXd &initial_, BoundaryFN boundary_conditions_, std::string name_) :
        mesh{mesh_}, current{initial_}, boundary_conditions{std::move(boundary_conditions_)}, name{std::move(name_)} {
    num_nodes = mesh->_num_nodes;
    is_basically_created = true;
}

Variable::Variable(const std::shared_ptr<Variable> left_operand_, const std::shared_ptr<Variable> right_operand_,
                   std::function<Eigen::MatrixXd(Eigen::MatrixXd &, Eigen::MatrixXd &)> op_, std::string &name_) :
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
//    auto* new_obj = new DT(*this);
//    return new_obj;
//    return (Variable *) this;
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

std::shared_ptr<Variable> _Stab::clone() const  {
    auto* new_obj = new _Stab(*this);
    return std::shared_ptr<Variable>{new_obj};
}


//Variable::Variable(const Variable & var) {
//    name = var.name;
//    mesh = var.mesh;
//    current = var.current;
//    boundary_conditions = var.boundary_conditions;
//    history = var.history;
//    num_nodes = var.num_nodes;
//    is_subvariable = var.is_subvariable;
//    is_constvar = var.is_constvar;
//    op = var.op;
//
//    if (var.left_operand != nullptr) {
//        left_operand =
//    }
//}


void Variable::set_bound() {
    current = boundary_conditions(mesh, current);
}

void Variable::add_history() {
    history.push_back({current});
}

Eigen::MatrixXd Variable::estimate_grads() {
    if (estimate_grid_cache_valid) {
        return estimate_grid_cache;
    }

    estimate_grid_cache = Eigen::Matrix<double, -1, 2> { num_nodes, 2};
    estimate_grid_cache.setConstant(0);
    auto pre_ret = Eigen::Matrix<double, -1, 4> { num_nodes, 4};

    for (int i = 0; i < num_nodes; ++i) {
        auto& node = mesh->_nodes.at(i);
        for (int j = 0; j < node->_edges_id.size(); ++j) {
            auto edge_id = node->_edges_id.at(j);
            auto& edge = mesh->_edges.at(edge_id);

            auto n1_id = i, n2_id = i;
            if (edge->_nodes_id.size() > 1) {
                auto& n1_ = mesh->_nodes.at(edge->_nodes_id.at(0));
                auto& n2_ = mesh->_nodes.at(edge->_nodes_id.at(1));
                if (n1_->_id == i) {
                    n2_id = n2_->_id;
                } else {
                    n2_id = n1_->_id;
                }
            }

            auto fi = (current(n1_id) + current(n2_id)) / 2;
            pre_ret(i, j) = fi;
        }
    }

    Eigen::MatrixXd pre_ret_x = pre_ret.cwiseProduct(mesh->_normal_x);
    Eigen::MatrixXd pre_ret_y = pre_ret.cwiseProduct(mesh->_normal_y);

    Eigen::Matrix<double, -1, 1> grad_x = pre_ret_x.rowwise().sum();
    Eigen::Matrix<double, -1, 1> grad_y = pre_ret_y.rowwise().sum();

    grad_x = grad_x.cwiseQuotient(mesh->_volumes);
    grad_y = grad_y.cwiseQuotient(mesh->_volumes);

    estimate_grid_cache.col(0) = grad_x;
    estimate_grid_cache.col(1) = grad_y;
    estimate_grid_cache_valid = true;

    return estimate_grid_cache;
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
    auto grads_self_x = Eigen::Matrix<double, -1, 4> { num_nodes, 4};
    auto grads_self_y = Eigen::Matrix<double, -1, 4> { num_nodes, 4};
    auto grads_neigh_x = Eigen::Matrix<double, -1, 4> { num_nodes, 4};
    auto grads_neigh_y = Eigen::Matrix<double, -1, 4> { num_nodes, 4};
    auto cur_self = Eigen::Matrix<double, -1, 4> { num_nodes, 4};
    auto cur_neigh = Eigen::Matrix<double, -1, 4> { num_nodes, 4};

    auto ret_sum = Eigen::Matrix<double, -1, 4> { num_nodes, 4};
    auto ret_r = Eigen::Matrix<double, -1, 4> { num_nodes, 4};
    auto ret_l = Eigen::Matrix<double, -1, 4> { num_nodes, 4};


    for (int i = 0; i < num_nodes; ++i) {
        auto& node = mesh->_nodes.at(i);

        for (int j = 0; j < node->_edges_id.size(); ++j) {
            auto edge_id = node->_edges_id.at(j);
            auto& edge = mesh->_edges.at(edge_id);

            auto& n1_id = i, n2_id = i;
            if (edge->_nodes_id.size() > 1) {
                auto& n1_ = mesh->_nodes.at(edge->_nodes_id.at(0));
                auto& n2_ = mesh->_nodes.at(edge->_nodes_id.at(1));
                if (n1_->_id == i) {
                    n2_id = n2_->_id;
                } else {
                    n2_id = n1_->_id;
                }
            }
            grads_self_x(i, j) = grads(n1_id, 0);
            grads_self_y(i, j) = grads(n1_id, 1);
            grads_neigh_x(i, j) = grads(n2_id, 0);
            grads_neigh_y(i, j) = grads(n2_id, 1);
            cur_self(i, j) = current(n1_id);
            cur_neigh(i, j) = current(n2_id);
        }
    }

    grads_self_x = grads_self_x.cwiseProduct(mesh->_vec_in_edge_direction_x);
    grads_self_y = grads_self_y.cwiseProduct(mesh->_vec_in_edge_direction_y);
    grads_neigh_x = grads_neigh_x.cwiseProduct(mesh->_vec_in_edge_neigh_direction_x);
    grads_neigh_y = grads_neigh_y.cwiseProduct(mesh->_vec_in_edge_neigh_direction_y);

    auto val_self = grads_self_x + grads_self_y + cur_self;
    auto val_neigh = grads_neigh_x + grads_neigh_y + cur_neigh;

    ret_sum = (val_self + val_neigh) / 2;

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
    return {l_p, r_p, [](Eigen::MatrixXd& lft, Eigen::MatrixXd& rht){return lft + rht;}, name_};
}

Variable Variable::operator-(const Variable &obj_r) const {
    std::string name_ = "(" + this->name + "-" + obj_r.name + ")";
    auto [l_p, r_p] = get_that_vars(this, obj_r);
    return {l_p, r_p, [](Eigen::MatrixXd& lft, Eigen::MatrixXd& rht){return lft - rht;}, name_};
}

Variable Variable::operator*(const Variable &obj_r) const {
    std::string name_ = "(" + this->name + "*" + obj_r.name + ")";
    auto [l_p, r_p] = get_that_vars(this, obj_r);
    return {l_p, r_p, [](Eigen::MatrixXd& lft, Eigen::MatrixXd& rht){return lft.array() * rht.array();}, name_};
}

Variable Variable::operator/(const Variable &obj_r) const {
    std::string name_ = "(" + this->name + "/" + obj_r.name + ")";
    auto [l_p, r_p] = get_that_vars(this, obj_r);
    return {l_p, r_p, [](Eigen::MatrixXd& lft, Eigen::MatrixXd& rht){return lft.cwiseQuotient(rht);}, name_};
}

Variable Variable::operator-() const {
    std::string name_ = "-(" + this->name + ")";
    auto [l_p, r_p] = get_that_vars(this, *this);
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

Eigen::MatrixXd DT::evaluate() {
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
}

Eigen::MatrixXd _Grad::evaluate() {
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
}

_Stab::_Stab(Variable *var_, bool clc_x_, bool clc_y_) : clc_x{clc_x_}, clc_y{clc_y_} {
    var = std::shared_ptr<Variable>{var_->clone()};
}

Eigen::MatrixXd _Stab::evaluate() {
    auto current_interface_gradients = var->get_interface_vars_first_order();
    auto& current_interface_gradients_star = (std::get<2>(current_interface_gradients) - std::get<1>(current_interface_gradients)) / 2;

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
}

void EqSolver::solve_dt(Variable *equation, Variable *time_var, Variable *set_var, DT *dt) {
    Eigen::VectorXd current = equation->evaluate();
    auto extracted = time_var->extract(current, dt->_dt);
    set_var->set_current(extracted);
}

Equation::Equation(size_t timesteps_) : timesteps{timesteps_} {}

void Equation::evaluate(std::vector<Variable*> &all_vars,
                        std::vector<std::tuple<Variable*, char, Variable>> &equation_system, DT* dt) {
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

        double progress = (static_cast<double>(t) / static_cast<double>(timesteps)) * 100;
        bar.set_progress(static_cast<size_t>(progress));
        bar.set_option(indicators::option::PostfixText{std::to_string(progress) + " %"});
    }
}

Eigen::MatrixXd to_grid(Mesh2D* mesh, Eigen::VectorXd& values) {
    Eigen::MatrixXd result = {mesh->_x, mesh->_y};
    for (auto& node : mesh->_nodes) {
        double value = values(node->_id);
        size_t x_coord = node->_id / mesh->_x;
        size_t y_coord = node->_id % mesh->_x;
        result(x_coord, y_coord) = value;
    }
    return result;
}
