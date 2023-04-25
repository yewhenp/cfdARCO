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
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
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
    gpuErrchk( cudaPeekAtLastError() );
    for (int i = 0; i < cols; i++) {
        from_multiple_k<<<nblocks, blocksize>>>(a[i].data.get(), res.data.get(), rows, cols, i);
        gpuErrchk( cudaPeekAtLastError() );
    }
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );

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
    cudaDeviceSynchronize();

    return res;
}

