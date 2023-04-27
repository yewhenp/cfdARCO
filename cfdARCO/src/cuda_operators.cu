#include "cuda_operators.hpp"
#include "decls.hpp"
#include "Eigen/Dense"

#define BLOCK_SIZE 1024

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void add_mtrx_k(const double* a, const double* b, double* c, int n) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

CudaDataMatrix add_mtrx(const CudaDataMatrix& a, const CudaDataMatrix& b) {
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<double>(a._size) / static_cast<double>(blocksize));
    CudaDataMatrix res {a._size};
    add_mtrx_k<<<nblocks, blocksize>>>(a.data.get(), b.data.get(), res.data.get(), a._size);
//    cudaDeviceSynchronize();
    return res;
}

__global__ void sub_mtrx_k(const double* a, const double* b, double* c, int n) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

CudaDataMatrix sub_mtrx(const CudaDataMatrix& a, const CudaDataMatrix& b) {
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<double>(a._size) / static_cast<double>(blocksize));
    CudaDataMatrix res {a._size};
    sub_mtrx_k<<<nblocks, blocksize>>>(a.data.get(), b.data.get(), res.data.get(), a._size);
//    cudaDeviceSynchronize();
    return res;
}

__global__ void mul_mtrx_k(const double* a, const double* b, double* c, int n) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

CudaDataMatrix mul_mtrx(const CudaDataMatrix& a, const CudaDataMatrix& b) {
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<double>(a._size) / static_cast<double>(blocksize));
    CudaDataMatrix res {a._size};
    mul_mtrx_k<<<nblocks, blocksize>>>(a.data.get(), b.data.get(), res.data.get(), a._size);
//    cudaDeviceSynchronize();
    return res;
}

__global__ void mul_mtrx_rowwice_k(const double* a, const double* b, double* c, int rows, int cols) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b_idx = (idx - (idx % cols)) / cols;
    if(idx < rows * cols) {
        c[idx] = a[idx] * b[b_idx];
    }
}

CudaDataMatrix mul_mtrx_rowwice(const CudaDataMatrix& a, const CudaDataMatrix& b, int rows, int cols) {
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<double>(a._size) / static_cast<double>(blocksize));
    CudaDataMatrix res {a._size};
    mul_mtrx_rowwice_k<<<nblocks, blocksize>>>(a.data.get(), b.data.get(), res.data.get(), rows, cols);
//    cudaDeviceSynchronize();
    return res;
}

__global__ void mul_mtrx_rowjump_k(const double* a, const double* b, double* c, int rows, int cols, int col_id) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    int a_idx = idx * cols + col_id;
    if(idx < rows) {
        c[idx] = b[idx] * a[a_idx];
    }
}

CudaDataMatrix mul_mtrx_rowjump(const CudaDataMatrix& a, const CudaDataMatrix& b, int rows, int cols, int col_id) {
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<double>(b._size) / static_cast<double>(blocksize));
    CudaDataMatrix res {b._size};
    mul_mtrx_rowjump_k<<<nblocks, blocksize>>>(a.data.get(), b.data.get(), res.data.get(), rows, cols, col_id);
//    cudaDeviceSynchronize();
    return res;
}

__global__ void div_mtrx_k(const double* a, const double* b, double* c, int n) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        c[idx] = a[idx] / b[idx];
    }
}

CudaDataMatrix div_mtrx(const CudaDataMatrix& a, const CudaDataMatrix& b) {
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<double>(a._size) / static_cast<double>(blocksize));
    CudaDataMatrix res {a._size};
    div_mtrx_k<<<nblocks, blocksize>>>(a.data.get(), b.data.get(), res.data.get(), a._size);
//    cudaDeviceSynchronize();
    return res;
}

__global__ void div_const_k(const double* a, const double b, double* c, int n) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        c[idx] = a[idx] / b;
    }
}

CudaDataMatrix div_const(const CudaDataMatrix& a, const double b) {
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<double>(a._size) / static_cast<double>(blocksize));
    CudaDataMatrix res {a._size};
    div_const_k<<<nblocks, blocksize>>>(a.data.get(), b, res.data.get(), a._size);
//    cudaDeviceSynchronize();
    return res;
}

__global__ void neg_mtrx_k(const double* a, double* c, int n) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        c[idx] = -a[idx];
    }
}

CudaDataMatrix neg_mtrx(const CudaDataMatrix& a) {
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<double>(a._size) / static_cast<double>(blocksize));
    CudaDataMatrix res {a._size};
    neg_mtrx_k<<<nblocks, blocksize>>>(a.data.get(), res.data.get(), a._size);
//    cudaDeviceSynchronize();
    return res;
}


__global__ void rowwice_sum_k(const double* a, double* c, int rows, int cols) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < rows) {
        for (int i = 0; i < cols; ++i) {
            c[idx] = c[idx] + a[idx * cols + i];
        }
    }
}

CudaDataMatrix rowwice_sum(const CudaDataMatrix& a, int rows, int cols) {
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<double>(rows) / static_cast<double>(blocksize));
    CudaDataMatrix res {rows, 0};
    rowwice_sum_k<<<nblocks, blocksize>>>(a.data.get(), res.data.get(), rows, cols);
//    cudaDeviceSynchronize();
    return res;
}


__global__ void from_multiple_k(const double* a, double* c, int rows, int cols, int col_id) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < rows) {
        c[idx * cols + col_id] = a[idx];
    }
}

CudaDataMatrix from_multiple_cols(const std::vector<CudaDataMatrix>& a) {
    auto rows = a.at(0)._size;
    auto cols = a.size();
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<double>(rows) / static_cast<double>(blocksize));
    CudaDataMatrix res {rows * cols};
    for (int i = 0; i < cols; i++) {
        from_multiple_k<<<nblocks, blocksize>>>(a[i].data.get(), res.data.get(), rows, cols, i);
    }
//    cudaDeviceSynchronize();

    return res;
}


__global__ void get_col_k(const double* a, double* c, int rows, int cols, int col_id) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < rows) {
        c[idx] = a[idx * cols + col_id];
    }
}

CudaDataMatrix get_col(const CudaDataMatrix& a, int rows, int cols, int col_id) {
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<double>(rows) / static_cast<double>(blocksize));
    CudaDataMatrix res {rows};
    get_col_k<<<nblocks, blocksize>>>(a.data.get(), res.data.get(), rows, cols, col_id);
//    cudaDeviceSynchronize();

    return res;
}


__global__ void estimate_grads_k(
        double* current_redist_cu1,
        double* current_redist_cu2,
        double* current_redist_cu3,
        double* current_redist_cu4,
        double* current_cu,
        double* normal_x_cu,
        double* normal_y_cu,
        double* volumes_cu,
        double* grad_x,
        double* grad_y,
        int rows
        ) {

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto t = threadIdx.x;

//    __shared__ double sm[4096];
    double curr_val, x_accum, y_accum;

    int idx_norm_1 = idx * 4 + 0;
    int idx_norm_2 = idx * 4 + 1;
    int idx_norm_3 = idx * 4 + 2;
    int idx_norm_4 = idx * 4 + 3;

    if(idx < rows) {
        double current_cu_idx = current_cu[idx];

        curr_val = (current_cu_idx + current_redist_cu1[idx]) / 2.;
        x_accum = 0. + curr_val * normal_x_cu[idx_norm_1];
        y_accum = 0. + curr_val * normal_y_cu[idx_norm_1];

        curr_val = (current_cu_idx + current_redist_cu2[idx]) / 2.;
        x_accum = x_accum + curr_val * normal_x_cu[idx_norm_2];
        y_accum = y_accum + curr_val * normal_y_cu[idx_norm_2];

        curr_val = (current_cu_idx + current_redist_cu3[idx]) / 2.;
        x_accum = x_accum + curr_val * normal_x_cu[idx_norm_3];
        y_accum = y_accum + curr_val * normal_y_cu[idx_norm_3];

        curr_val = (current_cu_idx + current_redist_cu4[idx]) / 2.;
        x_accum = x_accum + curr_val * normal_x_cu[idx_norm_4];
        y_accum = y_accum + curr_val * normal_y_cu[idx_norm_4];

        grad_x[idx] = x_accum / volumes_cu[idx];
        grad_y[idx] = y_accum / volumes_cu[idx];
    }
}

void estimate_grads_kern(const std::vector<CudaDataMatrix>& current_redist_cu,
                                   const CudaDataMatrix& current_cu,
                                   const CudaDataMatrix& normal_x_cu,
                                   const CudaDataMatrix& normal_y_cu,
                                   const CudaDataMatrix& volumes_cu,
                                   CudaDataMatrix& grad_x,
                                   CudaDataMatrix& grad_y
                                   ) {
    int rows = current_cu._size;
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<double>(rows) / static_cast<double>(blocksize));
;
    estimate_grads_k<<<nblocks, blocksize>>>(
            current_redist_cu[0].data.get(),
            current_redist_cu[1].data.get(),
            current_redist_cu[2].data.get(),
            current_redist_cu[3].data.get(),
            current_cu.data.get(),
            normal_x_cu.data.get(),
            normal_y_cu.data.get(),
            volumes_cu.data.get(),
            grad_x.data.get(),
            grad_y.data.get(),
            rows
            );
}


__global__ void get_interface_vars_first_order_k(
        double* grads_x,
        double* grads_y,
        double* grad_redist_cu_x1,
        double* grad_redist_cu_x2,
        double* grad_redist_cu_x3,
        double* grad_redist_cu_x4,
        double* grad_redist_cu_y1,
        double* grad_redist_cu_y2,
        double* grad_redist_cu_y3,
        double* grad_redist_cu_y4,
        double* current_cu,
        double* current_redist_cu1,
        double* current_redist_cu2,
        double* current_redist_cu3,
        double* current_redist_cu4,
        double* vec_in_edge_direction_x_cu,
        double* vec_in_edge_direction_y_cu,
        double* vec_in_edge_neigh_direction_x_cu,
        double* vec_in_edge_neigh_direction_y_cu,
        double* ret_sum_cu,
        double* ret_r_cu,
        double* ret_l_cu,
        int rows
) {
    const int cols = 4;
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col_id = idx % cols;
    int row_id = (idx - col_id) / cols;

    double val_self_cu = 0;
    double val_neigh_cu = 0;

    double* grad_redist_cu_x;
    double* grad_redist_cu_y;
    double* current_redist_cu;
    if (col_id == 0) {
        grad_redist_cu_x = grad_redist_cu_x1;
        grad_redist_cu_y = grad_redist_cu_y1;
        current_redist_cu = current_redist_cu1;
    } else if (col_id == 1) {
        grad_redist_cu_x = grad_redist_cu_x2;
        grad_redist_cu_y = grad_redist_cu_y2;
        current_redist_cu = current_redist_cu2;
    } else if (col_id == 2) {
        grad_redist_cu_x = grad_redist_cu_x3;
        grad_redist_cu_y = grad_redist_cu_y3;
        current_redist_cu = current_redist_cu3;
    } else {
        grad_redist_cu_x = grad_redist_cu_x4;
        grad_redist_cu_y = grad_redist_cu_y4;
        current_redist_cu = current_redist_cu4;
    }

    if(idx < rows * 4) {
        val_self_cu = grads_x[row_id] * vec_in_edge_direction_x_cu[idx];
        val_self_cu = val_self_cu + grads_y[row_id] * vec_in_edge_direction_y_cu[idx];
        val_self_cu = val_self_cu + current_cu[row_id];

        val_neigh_cu = grad_redist_cu_x[row_id] * vec_in_edge_neigh_direction_x_cu[idx];
        val_neigh_cu = val_neigh_cu + grad_redist_cu_y[row_id] * vec_in_edge_neigh_direction_y_cu[idx];
        val_neigh_cu = val_neigh_cu + current_redist_cu[row_id];

        ret_sum_cu[idx] = (val_self_cu + val_neigh_cu) / 2.;

        if (col_id == 0 || col_id == 3) {
            ret_r_cu[idx] = val_neigh_cu;
            ret_l_cu[idx] = val_self_cu;
        } else {
            ret_r_cu[idx] = val_self_cu;
            ret_l_cu[idx] = val_neigh_cu;
        }
    }
}

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
) {
    int rows = current_cu._size;
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<double>(rows * 4) / static_cast<double>(blocksize));

    get_interface_vars_first_order_k<<<nblocks, blocksize>>>(
            grad_x.data.get(),
            grad_y.data.get(),
            std::get<0>(grad_redist_cu.at(0)).data.get(),
            std::get<0>(grad_redist_cu.at(1)).data.get(),
            std::get<0>(grad_redist_cu.at(2)).data.get(),
            std::get<0>(grad_redist_cu.at(3)).data.get(),
            std::get<1>(grad_redist_cu.at(0)).data.get(),
            std::get<1>(grad_redist_cu.at(1)).data.get(),
            std::get<1>(grad_redist_cu.at(2)).data.get(),
            std::get<1>(grad_redist_cu.at(3)).data.get(),
            current_cu.data.get(),
            current_redist_cu.at(0).data.get(),
            current_redist_cu.at(1).data.get(),
            current_redist_cu.at(2).data.get(),
            current_redist_cu.at(3).data.get(),
            vec_in_edge_direction_x_cu.data.get(),
            vec_in_edge_direction_y_cu.data.get(),
            vec_in_edge_neigh_direction_x_cu.data.get(),
            vec_in_edge_neigh_direction_y_cu.data.get(),
            ret_sum_cu.data.get(),
            ret_r_cu.data.get(),
            ret_l_cu.data.get(),
            rows
    );
}


//__global__ void _Grad_k(
//        double* current_redist_cu1,
//        double* current_redist_cu2,
//        double* current_redist_cu3,
//        double* current_redist_cu4,
//        double* current_cu,
//        double* normal_x_cu,
//        double* normal_y_cu,
//        double* volumes_cu,
//        double* grad_x,
//        double* grad_y,
//        int rows
//) {
//
//    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
//    auto t = threadIdx.x;
//
////    __shared__ double sm[4096];
//    double curr_val, x_accum, y_accum;
//
//    int idx_norm_1 = idx * 4 + 0;
//    int idx_norm_2 = idx * 4 + 1;
//    int idx_norm_3 = idx * 4 + 2;
//    int idx_norm_4 = idx * 4 + 3;
//
//    if(idx < rows) {
//        double current_cu_idx = current_cu[idx];
//
//        curr_val = (current_cu_idx + current_redist_cu1[idx]) / 2.;
//        x_accum = 0. + curr_val * normal_x_cu[idx_norm_1];
//        y_accum = 0. + curr_val * normal_y_cu[idx_norm_1];
//
//        curr_val = (current_cu_idx + current_redist_cu2[idx]) / 2.;
//        x_accum = x_accum + curr_val * normal_x_cu[idx_norm_2];
//        y_accum = y_accum + curr_val * normal_y_cu[idx_norm_2];
//
//        curr_val = (current_cu_idx + current_redist_cu3[idx]) / 2.;
//        x_accum = x_accum + curr_val * normal_x_cu[idx_norm_3];
//        y_accum = y_accum + curr_val * normal_y_cu[idx_norm_3];
//
//        curr_val = (current_cu_idx + current_redist_cu4[idx]) / 2.;
//        x_accum = x_accum + curr_val * normal_x_cu[idx_norm_4];
//        y_accum = y_accum + curr_val * normal_y_cu[idx_norm_4];
//
//        grad_x[idx] = x_accum / volumes_cu[idx];
//        grad_y[idx] = y_accum / volumes_cu[idx];
//    }
//}
//
//void estimate_grads_kern(const std::vector<CudaDataMatrix>& current_redist_cu,
//                         const CudaDataMatrix& current_cu,
//                         const CudaDataMatrix& normal_x_cu,
//                         const CudaDataMatrix& normal_y_cu,
//                         const CudaDataMatrix& volumes_cu,
//                         CudaDataMatrix& grad_x,
//                         CudaDataMatrix& grad_y
//) {
//    int rows = current_cu._size;
//    int blocksize = BLOCK_SIZE;
//    int nblocks = std::ceil(static_cast<double>(rows) / static_cast<double>(blocksize));
//    ;
//    estimate_grads_k<<<nblocks, blocksize>>>(
//            current_redist_cu[0].data.get(),
//            current_redist_cu[1].data.get(),
//            current_redist_cu[2].data.get(),
//            current_redist_cu[3].data.get(),
//            current_cu.data.get(),
//            normal_x_cu.data.get(),
//            normal_y_cu.data.get(),
//            volumes_cu.data.get(),
//            grad_x.data.get(),
//            grad_y.data.get(),
//            rows
//    );
//}
