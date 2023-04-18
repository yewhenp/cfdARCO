#ifndef CFDARCO_FVM_HPP
#define CFDARCO_FVM_HPP

#include "mesh2d.hpp"
#include "decls.hpp"
#include <optional>
#include <memory>

class Variable;
//class _SubVariable;
class DT;
class _GradEstimated;

using BoundaryFN = std::function<Eigen::VectorXd(Mesh2D* mesh, Eigen::VectorXd& arr)>;

class Variable {
public:
    Variable();
    Variable(Mesh2D* mesh_, Eigen::VectorXd& initial_, BoundaryFN boundary_conditions_, std::string name_="");
    Variable(const std::shared_ptr<Variable> left_operand_, const std::shared_ptr<Variable> right_operand_, std::function<MatrixX4dRB(MatrixX4dRB&, MatrixX4dRB&)> op_, std::string& name_);
    Variable(Mesh2D* mesh_, double value);
    Variable(Eigen::VectorXd& curr_);

    Variable(Variable&);
    Variable(const Variable&);
    Variable(Variable&&) = delete;
    Variable(const Variable&&) = delete;

    virtual std::shared_ptr<Variable> clone() const;

    void set_bound();
    void add_history();
    MatrixX4dRB estimate_grads();
    _GradEstimated dx();
    _GradEstimated dy();
    std::tuple<MatrixX4dRB, MatrixX4dRB, MatrixX4dRB> get_interface_vars_first_order();
    std::tuple<MatrixX4dRB, MatrixX4dRB, MatrixX4dRB> get_interface_vars_second_order();
    virtual Eigen::VectorXd extract(Eigen::VectorXd& left_part, double dt);
    virtual MatrixX4dRB evaluate();
    void set_current(Eigen::VectorXd& current_);
    std::vector<Eigen::VectorXd> get_history();
    virtual void solve(Variable* equation, DT* dt);

public:
    std::string name;
    Mesh2D *mesh = nullptr;
    Eigen::VectorXd current;
    std::vector<MatrixX4dRB> current_redist;
    std::vector<MatrixX4dRB> grad_redist;
    BoundaryFN boundary_conditions;
    std::vector<Eigen::VectorXd> history {};
    size_t num_nodes = 0;
    bool is_subvariable = false;
    bool is_constvar = false;
    bool is_basically_created = false;

//    from subvariable
    std::shared_ptr<Variable> left_operand = nullptr;
    std::shared_ptr<Variable> right_operand = nullptr;
    std::function<MatrixX4dRB(MatrixX4dRB&, MatrixX4dRB&)> op;

//    cache
    bool estimate_grid_cache_valid = false;
    MatrixX4dRB estimate_grid_cache;

    Variable operator+(const Variable & obj_r) const;
    Variable operator-(const Variable & obj_r) const;
    Variable operator*(const Variable & obj_r) const;
    Variable operator/(const Variable & obj_r) const;
    Variable operator-() const;
};

Variable operator+(const double obj_l, const Variable & obj_r);
Variable operator-(const double obj_l, const Variable & obj_r);
Variable operator*(const double obj_l, const Variable & obj_r);
Variable operator/(const double obj_l, const Variable & obj_r);
Variable operator+(const Variable & obj_l, const double obj_r);
Variable operator-(const Variable & obj_l, const double obj_r);
Variable operator*(const Variable & obj_l, const double obj_r);
Variable operator/(const Variable & obj_l, const double obj_r);

class _GradEstimated : public Variable {
public:
    explicit _GradEstimated(Variable *var_, bool clc_x_=true, bool clc_y_=true);

    MatrixX4dRB evaluate() override;
    std::shared_ptr<Variable> clone() const override;

    Variable* var;
    bool clc_x;
    bool clc_y;
};


class UpdatePolicies {
public:
    static double CourantFriedrichsLewy(double CFL, std::vector<Variable*>& space_vars, Mesh2D* mesh);
};


class DT : public Variable {
public:
    DT(Mesh2D* mesh_, std::function<double(double, std::vector<Variable*>&, Mesh2D* mesh)> update_fn_, double CFL_, std::vector<Variable*>& space_vars_);
    void update();
    MatrixX4dRB evaluate() override;
    std::shared_ptr<Variable> clone() const override;

    std::function<double(double, std::vector<Variable*>&, Mesh2D* mesh)> update_fn;
    std::vector<Variable*>& space_vars;
    double _dt = 0.0;
    double CFL = 0.0;
};

class Variable2d : Variable {
public:
    using Variable::Variable;
    using Variable::current;
    using Variable::operator*;
    using Variable::operator+;
    using Variable::operator-;
    using Variable::operator/;
};


class _DT : public Variable {
public:
    _DT(Variable* var_, int);

    Eigen::VectorXd extract(Eigen::VectorXd& left_part, double dt) override;
    void solve(Variable* equation, DT* dt) override;
    std::shared_ptr<Variable> clone() const override;

    std::shared_ptr<Variable> var;
};


class _Grad : public Variable {
public:
    _Grad(Variable* var_, bool clc_x_=1, bool clc_y_=1);

    MatrixX4dRB evaluate() override;
    std::shared_ptr<Variable> clone() const override;

    std::shared_ptr<Variable> var;
    bool clc_x;
    bool clc_y;
};

class _Grad2 : public Variable {
public:
    _Grad2(Variable* var_, bool clc_x_=1, bool clc_y_=1);

    MatrixX4dRB evaluate() override;
    std::shared_ptr<Variable> clone() const override;

    std::shared_ptr<Variable> var;
    bool clc_x;
    bool clc_y;
};


class _Stab : public Variable {
public:
    _Stab(Variable* var_, bool clc_x_=1, bool clc_y_=1);

    MatrixX4dRB evaluate() override;
    std::shared_ptr<Variable> clone() const override;

    std::shared_ptr<Variable> var;
    bool clc_x;
    bool clc_y;
};


inline auto d1t(Variable& var) {
    auto varr = new _DT(&var, 0);
    varr->name = "d1t(" + var.name + ")";
    return varr;
}

inline auto d1t(Variable&& var) {
    auto varr = new _DT(&var, 0);
    varr->name = "d1t(" + var.name + ")";
    return varr;
}

inline auto d1dx(Variable& var) {
    return _Grad(&var, true, false);
}

inline auto d1dx(Variable&& var) {
    return _Grad(&var, true, false);
}

inline auto d1dy(Variable& var) {
    return _Grad(&var,  false, true);
}

inline auto d1dy(Variable&& var) {
    return _Grad(&var,  false, true);
}

inline auto d2dx(Variable& var) {
    return _Grad2(&var, true, false);
}

inline auto d2dx(Variable&& var) {
    return _Grad2(&var, true, false);
}

inline auto d2dy(Variable& var) {
    return _Grad2(&var,  false, true);
}

inline auto d2dy2(Variable&& var) {
    return _Grad2(&var,  false, true);
}

inline auto stab_x(Variable& var) {
    return _Stab(&var, true, false);
}

inline auto stab_x(Variable&& var) {
    return _Stab(&var, true, false);
}

inline auto stab_y(Variable& var) {
    return _Stab(&var,  false, true);
}

inline auto stab_y(Variable&& var) {
    return _Stab(&var,  false, true);
}


class EqSolver {
public:
    static void solve_dt(Variable* equation, Variable* time_var, Variable* set_var, DT* dt);
};


class Equation {
public:
    Equation(size_t timesteps_);
    void evaluate(std::vector<Variable*>&all_vars, std::vector<std::tuple<Variable*, char, Variable>>&equation_system, DT* dt, bool visualize);

    size_t timesteps;
};


MatrixX4dRB to_grid(Mesh2D* mesh, Eigen::VectorXd& values);

template<typename Scalar, typename Matrix>
inline static std::vector< std::vector<Scalar> > from_eigen_matrix( const Matrix & M ){
    std::vector< std::vector<Scalar> > m;
    m.resize(M.rows(), std::vector<Scalar>(M.cols(), 0));
    for(size_t i = 0; i < m.size(); i++)
        for(size_t j = 0; j < m.front().size(); j++)
            m[i][j] = M(i,j);
    return m;
}


#endif //CFDARCO_FVM_HPP
