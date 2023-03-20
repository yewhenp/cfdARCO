#ifndef CFDARCO_FVM_HPP
#define CFDARCO_FVM_HPP

#include "mesh2d.hpp"
#include <optional>

class Variable;
//class _SubVariable;
class DT;
class _GradEstimated;

using BoundaryFN = std::function<Eigen::VectorXd(Mesh2D* mesh, Eigen::VectorXd& arr)>;

class Variable {
public:
    Variable();
    Variable(Mesh2D* mesh_, Eigen::VectorXd& initial_, BoundaryFN boundary_conditions_, std::string name_="");
    Variable(Variable* left_operand_, Variable* right_operand_, std::function<Eigen::MatrixXd(Eigen::MatrixXd&, Eigen::MatrixXd&)> op_, std::string& name_);
    void set_bound();
    void add_history();
    Eigen::MatrixXd estimate_grads();
    _GradEstimated dx();
    _GradEstimated dy();
    std::tuple<Eigen::MatrixX4d, Eigen::MatrixX4d, Eigen::MatrixX4d> get_interface_vars_first_order();
    Eigen::VectorXd extract(Eigen::VectorXd& left_part, double dt);
    virtual Eigen::VectorXd evaluate();
    void set_current(Eigen::VectorXd& current_);
    std::vector<Eigen::VectorXd> get_history();
    void solve(Variable& equation, DT& dt);


    std::string name;
    Mesh2D *mesh = nullptr;
    Eigen::VectorXd current;
    BoundaryFN boundary_conditions;
    std::vector<Eigen::VectorXd> history {};
    size_t num_nodes;
    bool is_subvariable = false;

//    from subvariable
    Variable* left_operand = nullptr;
    Variable* right_operand = nullptr;
    std::function<Eigen::MatrixXd(Eigen::MatrixXd&, Eigen::MatrixXd&)> op;

    Variable operator+(Variable & obj_r);
    Variable operator-(Variable & obj_r);
    Variable operator*(Variable & obj_r);
    Variable operator/(Variable & obj_r);
    Variable operator-();
};



class _GradEstimated : Variable {
public:
    explicit _GradEstimated(Variable* var_, bool clc_x_=true, bool clc_y_=true);

    Eigen::VectorXd evaluate() override;

    Variable* var;
    bool clc_x;
    bool clc_y;
};


class UpdatePolicies {
public:
    static double CourantFriedrichsLewy(double CFL, std::vector<Variable*>& space_vars);
};


class DT : Variable {
public:
    DT(Mesh2D* mesh_, std::function<double(double, std::vector<Variable*>&)> update_fn_, double CFL_, std::vector<Variable*>& space_vars_);
    void update();
    Eigen::VectorXd evaluate() override;

    Mesh2D* mesh = nullptr;
    std::function<double(double, std::vector<Variable*>&)> update_fn;
    std::vector<Variable*>& space_vars;
    double _dt = 0.0;
    double CFL = 0.0;
};

class Variable2d : Variable {};

//class _SubVariable : Variable {
//public:
//    _SubVariable(Variable& left_operand_, Variable& right_operand_, std::function<Eigen::MatrixXd(Eigen::MatrixXd&, Eigen::MatrixXd&)> op_, std::string& name_);
//
//    Variable* left_operand;
//    Variable* right_operand;
//    std::function<Eigen::MatrixXd(Eigen::MatrixXd&, Eigen::MatrixXd&)> op;
//    std::string& name;
//};

class _DT : Variable {
public:
    _DT(Variable* var_);

    Variable* var;
};


class _Grad : Variable {
public:
    _Grad(Variable* var_, bool clc_x_=1, bool clc_y_=1);

    Variable* var;
    bool clc_x;
    bool clc_y;
};


class _Stab : Variable {
public:
    _Stab(Variable* var_, bool clc_x_=1, bool clc_y_=1);

    Variable* var;
    bool clc_x;
    bool clc_y;
};


inline auto d1t(Variable* var) {
    return _DT(var);
}

inline auto d1dx(Variable* var) {
    return _Grad(var, true, false);
}

inline auto d1dy(Variable* var) {
    return _Grad(var,  false, true);
}

inline auto stab_x(Variable* var) {
    return _Stab(var, true, false);
}

inline auto stab_y(Variable* var) {
    return _Stab(var,  false, true);
}


class EqSolver {
public:
    static void solve_dt(Variable* equation, Variable* time_var, Variable* set_var, DT* dt);
};


class Equation {
public:
    Equation(size_t timesteps_);
    void evaluate(std::vector<Variable*>&all_vars, std::vector<std::tuple<Variable*, char, Variable*>>&equation_system, DT* dt);

    size_t timesteps;
};



#endif //CFDARCO_FVM_HPP
