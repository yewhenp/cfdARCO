#ifndef CFDARCO_ABSTRACT_MESH_HPP
#define CFDARCO_ABSTRACT_MESH_HPP


#include <Eigen/Core>

class AbstractVertex {
public:
    virtual void compute() = 0;
    [[nodiscard]] virtual Eigen::VectorXd coordinates() const = 0;
};

class AbstractEdge {
public:
    virtual void compute() = 0;
    [[nodiscard]] virtual bool is_boundary() const = 0;
};

class AbstractCell {
public:
    virtual void compute() = 0;
    [[nodiscard]] virtual Eigen::VectorXd center_coords() const = 0;
    [[nodiscard]] virtual bool is_boundary() const = 0;
};

class AbstractMesh {
public:
    virtual void compute() = 0;
};

#endif //CFDARCO_ABSTRACT_MESH_HPP
