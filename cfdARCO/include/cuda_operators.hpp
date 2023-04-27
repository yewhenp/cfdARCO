#ifndef CFDARCO_CUDA_OPERATORS_HPP
#define CFDARCO_CUDA_OPERATORS_HPP

#ifdef CFDARCHO_CUDA_ENABLE

#include <memory>
#include <new>
#include <cuda_runtime_api.h>
#include "pool_allocator.hpp"
#include "decls.hpp"
#include "pool_allocator.hpp"

class CudaDeleter {
public:
    CudaDeleter(size_t size): _size{size} {}

    void operator ()( void* p)
    {
        if (_size > 0 && Allocator::allocator_alive) {
            Allocator::cuda_mem_pool->deallocate(p, _size * sizeof(double));
        }
    }

    size_t _size;
};

class CudaDataMatrix {
public:
    CudaDataMatrix() {
        data = std::shared_ptr<double> (nullptr, CudaDeleter{0});
    }

    explicit CudaDataMatrix(size_t size): _size{size} {
        auto* ptr = Allocator::cuda_mem_pool->allocate(_size * sizeof(double));
        data = std::shared_ptr<double> (static_cast<double*>(ptr), CudaDeleter{size});
    }

    CudaDataMatrix(size_t size, double const_val): _size{size} {
        auto* ptr = Allocator::cuda_mem_pool->allocate(_size * sizeof(double));
        data = std::shared_ptr<double> (static_cast<double*>(ptr), CudaDeleter{size});

        if (std::abs(const_val) <= 0.000001) {
            cudaMemset(data.get(), 0, _size * sizeof(double));
        } else {
            std::vector<double> copy_mem(size, const_val);
            cudaMemcpy(data.get(), copy_mem.data(), _size * sizeof(double), cudaMemcpyHostToDevice);
        }
    }

    MatrixX4dRB to_eigen(int rows, int cols) {
        MatrixX4dRB ret {rows, cols};
        cudaMemcpy(ret.data(), data.get(), _size * sizeof(double), cudaMemcpyDeviceToHost);
        return ret;
    }

    static CudaDataMatrix from_eigen(MatrixX4dRB mtrx) {
        auto mtrx_size = mtrx.size();
        CudaDataMatrix ret {static_cast<size_t>(mtrx_size)};
        cudaMemcpy(ret.data.get(), mtrx.data(), mtrx_size * sizeof(double), cudaMemcpyHostToDevice);
        return ret;
    }

    size_t _size;
    std::shared_ptr<double> data;
};

CudaDataMatrix add_mtrx(const CudaDataMatrix& a, const CudaDataMatrix& b);
CudaDataMatrix sub_mtrx(const CudaDataMatrix& a, const CudaDataMatrix& b);
CudaDataMatrix mul_mtrx(const CudaDataMatrix& a, const CudaDataMatrix& b);
CudaDataMatrix div_mtrx(const CudaDataMatrix& a, const CudaDataMatrix& b);
CudaDataMatrix neg_mtrx(const CudaDataMatrix& a);
CudaDataMatrix rowwice_sum(const CudaDataMatrix& a, int rows, int cols);
CudaDataMatrix mul_mtrx_rowwice(const CudaDataMatrix& a, const CudaDataMatrix& b, int rows, int cols);
CudaDataMatrix mul_mtrx_rowjump(const CudaDataMatrix& a, const CudaDataMatrix& b, int rows, int cols, int col_id);
CudaDataMatrix from_multiple_cols(const std::vector<CudaDataMatrix>& a);
CudaDataMatrix get_col(const CudaDataMatrix& a, int rows, int cols, int col_id);
CudaDataMatrix div_const(const CudaDataMatrix& a, const double b);
void estimate_grads_kern(const std::vector<CudaDataMatrix>& current_redist_cu,
                                   const CudaDataMatrix& current_cu,
                                   const CudaDataMatrix& normal_x_cu,
                                   const CudaDataMatrix& normal_y_cu,
                                   const CudaDataMatrix& volumes_cu,
                                   CudaDataMatrix& grad_x,
                                   CudaDataMatrix& grad_y
);
void get_interface_vars_first_order_kern(
        const CudaDataMatrix& grad_x,
        const CudaDataMatrix& grad_y,
        const std::vector<std::tuple<CudaDataMatrix, CudaDataMatrix>>& grad_redist_cu,
        const CudaDataMatrix& current_cu,
        const std::vector<CudaDataMatrix>& current_redist_cu,
        const CudaDataMatrix& vec_in_edge_direction_x_cu,
        const CudaDataMatrix& vec_in_edge_direction_y_cu,
        const CudaDataMatrix& vec_in_edge_neigh_direction_x_cu,
        const CudaDataMatrix& vec_in_edge_neigh_direction_y_cu,
        CudaDataMatrix& ret_sum_cu,
        CudaDataMatrix& ret_r_cu,
        CudaDataMatrix& ret_l_cu
);

inline CudaDataMatrix operator+(const CudaDataMatrix& obj_l, const CudaDataMatrix& obj_r) {
    return add_mtrx(obj_l, obj_r);
}

inline CudaDataMatrix operator-(const CudaDataMatrix& obj_l, const CudaDataMatrix& obj_r) {
    return sub_mtrx(obj_l, obj_r);
}

inline CudaDataMatrix operator*(const CudaDataMatrix& obj_l, const CudaDataMatrix& obj_r) {
    return mul_mtrx(obj_l, obj_r);
}

inline CudaDataMatrix operator/(const CudaDataMatrix& obj_l, const CudaDataMatrix& obj_r) {
    return div_mtrx(obj_l, obj_r);
}

inline CudaDataMatrix operator-(const CudaDataMatrix& obj_l) {
    return neg_mtrx(obj_l);
}

inline void sync_device() {
    cudaDeviceSynchronize();
}

#else

#include <memory>

class CudaDeleter {
public:
    CudaDeleter(size_t size) {  }
    void operator ()( void* p) { throw std::runtime_error{"CUDA not available"}; }
};

class CudaDataMatrix {
public:
    CudaDataMatrix() {  }
    explicit CudaDataMatrix(size_t size) { throw std::runtime_error{"CUDA not available"}; }
    CudaDataMatrix(size_t size, double const_val)  {throw std::runtime_error{"CUDA not available"}; }
    MatrixX4dRB to_eigen(int rows, int cols) { throw std::runtime_error{"CUDA not available"}; }
    static CudaDataMatrix from_eigen(MatrixX4dRB mtrx) { throw std::runtime_error{"CUDA not available"}; }

    size_t _size;
    std::shared_ptr<double> data;
};

inline CudaDataMatrix add_mtrx(const CudaDataMatrix& a, const CudaDataMatrix& b) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrix sub_mtrx(const CudaDataMatrix& a, const CudaDataMatrix& b) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrix mul_mtrx(const CudaDataMatrix& a, const CudaDataMatrix& b) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrix div_mtrx(const CudaDataMatrix& a, const CudaDataMatrix& b) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrix neg_mtrx(const CudaDataMatrix& a) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrix rowwice_sum(const CudaDataMatrix& a, int rows, int cols) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrix mul_mtrx_rowwice(const CudaDataMatrix& a, const CudaDataMatrix& b, int rows, int cols) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrix mul_mtrx_rowjump(const CudaDataMatrix& a, const CudaDataMatrix& b, int rows, int cols, int col_id) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrix from_multiple_cols(const std::vector<CudaDataMatrix>& a) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrix get_col(const CudaDataMatrix& a, int rows, int cols, int col_id) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrix div_const(const CudaDataMatrix& a, const double b) {throw std::runtime_error{"CUDA not available"};}
inline void estimate_grads_kern(const std::vector<CudaDataMatrix>& current_redist_cu,
                         const CudaDataMatrix& current_cu,
                         const CudaDataMatrix& normal_x_cu,
                         const CudaDataMatrix& normal_y_cu,
                         const CudaDataMatrix& volumes_cu,
                         CudaDataMatrix& grad_x,
                         CudaDataMatrix& grad_y
) {throw std::runtime_error{"CUDA not available"};}
inline void get_interface_vars_first_order_kern(
        const CudaDataMatrix& grad_x,
        const CudaDataMatrix& grad_y,
        const std::vector<std::tuple<CudaDataMatrix, CudaDataMatrix>>& grad_redist_cu,
        const CudaDataMatrix& current_cu,
        const std::vector<CudaDataMatrix>& current_redist_cu,
        const CudaDataMatrix& vec_in_edge_direction_x_cu,
        const CudaDataMatrix& vec_in_edge_direction_y_cu,
        const CudaDataMatrix& vec_in_edge_neigh_direction_x_cu,
        const CudaDataMatrix& vec_in_edge_neigh_direction_y_cu,
        CudaDataMatrix& ret_sum_cu,
        CudaDataMatrix& ret_r_cu,
        CudaDataMatrix& ret_l_cu
) {throw std::runtime_error{"CUDA not available"};}

inline CudaDataMatrix operator+(const CudaDataMatrix& obj_l, const CudaDataMatrix& obj_r) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrix operator-(const CudaDataMatrix& obj_l, const CudaDataMatrix& obj_r) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrix operator*(const CudaDataMatrix& obj_l, const CudaDataMatrix& obj_r) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrix operator/(const CudaDataMatrix& obj_l, const CudaDataMatrix& obj_r) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrix operator-(const CudaDataMatrix& obj_l) {throw std::runtime_error{"CUDA not available"};}
inline void sync_device() {}
#endif

#endif //CFDARCO_CUDA_OPERATORS_HPP
