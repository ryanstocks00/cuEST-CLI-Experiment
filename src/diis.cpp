/**
 * @file diis.cpp — Device Pulay DIIS (cuBLAS; history as n2×nvec matrices).
 */
#include "diis.hpp"

#include "cuest_wrapper/nvtx.hpp"

#include <cmath>
#include <cuda_runtime.h>
#include <vector>

namespace cuest {

void DIIS::init(int max_space, int N) {
  N_ = N;
  n2_ = N * N;
  max_space_ = max_space;
  count_ = 0;

  const size_t n2_bytes = static_cast<size_t>(n2_) * sizeof(double);
  const size_t hist_bytes = n2_bytes * static_cast<size_t>(max_space);
  d_E_.alloc(hist_bytes);
  d_F_.alloc(hist_bytes);
  d_gram_.alloc(static_cast<size_t>(max_space) * max_space * sizeof(double));
  d_coeffs_.alloc(static_cast<size_t>(max_space) * sizeof(double));
  d_tmp1_.alloc(n2_bytes);
  d_tmp2_.alloc(n2_bytes);
  d_err_.alloc(n2_bytes);
}

void DIIS::clear() { count_ = 0; }

void DIIS::push(cublasHandle_t blas, const double* d_err, const double* d_fock) {
  // Keep columns [0, count) packed oldest→newest for a single GEMM.
  if (count_ == max_space_) {
    for (int j = 0; j < max_space_ - 1; j++) {
      cublasDcopy(blas, n2_, err_col(j + 1), 1, err_col(j), 1);
      cublasDcopy(blas, n2_, fock_col(j + 1), 1, fock_col(j), 1);
    }
    count_ = max_space_ - 1;
  }
  cublasDcopy(blas, n2_, d_err, 1, err_col(count_), 1);
  cublasDcopy(blas, n2_, d_fock, 1, fock_col(count_), 1);
  ++count_;
}

void DIIS::extrapolate(cublasHandle_t blas, const double* d_S,
                       DeviceArray<double>& d_Fock, const double* d_D) {
  NvtxRange range("DIIS");
  const int N = N_;
  const int n2 = n2_;
  double one = 1.0, zero = 0.0, minus_one = -1.0;

  // FD = F * D
  cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one,
              d_Fock, N, d_D, N, &zero, d_tmp1_, N);
  // FDS = FD * S
  cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one,
              d_tmp1_, N, d_S, N, &zero, d_tmp2_, N);
  // SD = S * D
  cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one,
              d_S, N, d_D, N, &zero, d_tmp1_, N);
  // SDF = SD * F
  cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one,
              d_tmp1_, N, d_Fock, N, &zero, d_err_, N);
  // err = FDS - SDF
  cublasDcopy(blas, n2, d_tmp2_, 1, d_tmp1_, 1);
  cublasDaxpy(blas, n2, &minus_one, d_err_, 1, d_tmp1_, 1);
  cublasDcopy(blas, n2, d_tmp1_, 1, d_err_, 1);

  push(blas, d_err_, d_Fock);

  const int nvec = count_;
  if (nvec < 2) return;

  // Gram = Eᵀ E  (nvec × nvec), E is n2 × nvec column-major
  cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, nvec, nvec, n2, &one,
              d_E_, n2, d_E_, n2, &zero, d_gram_, nvec);

  std::vector<double> gram(static_cast<size_t>(nvec) * nvec);
  CUDA_CHECK(cudaMemcpy(gram.data(), d_gram_,
                        gram.size() * sizeof(double), cudaMemcpyDeviceToHost));

  // Augmented Pulay system: [Gram  -1; -1ᵀ  0] c = [0; -1]
  const int m = nvec + 1;
  std::vector<double> B(static_cast<size_t>(m) * m, 0.0);
  for (int j = 0; j < nvec; j++)
    for (int i = 0; i < nvec; i++)
      B[static_cast<size_t>(i + j * m)] = gram[static_cast<size_t>(i + j * nvec)];
  for (int i = 0; i < nvec; i++) {
    B[static_cast<size_t>(i + nvec * m)] = -1.0;
    B[static_cast<size_t>(nvec + i * m)] = -1.0;
  }
  std::vector<double> rhs(static_cast<size_t>(m), 0.0);
  rhs[static_cast<size_t>(nvec)] = -1.0;

  auto Bc = B;
  auto rhsc = rhs;
  bool ok = true;
  for (int col = 0; col < m; col++) {
    int pivot = col;
    for (int r = col + 1; r < m; r++)
      if (std::fabs(Bc[static_cast<size_t>(r + col * m)]) >
          std::fabs(Bc[static_cast<size_t>(pivot + col * m)]))
        pivot = r;
    if (std::fabs(Bc[static_cast<size_t>(pivot + col * m)]) < 1e-20) {
      ok = false;
      break;
    }
    if (pivot != col) {
      for (int j = 0; j < m; j++)
        std::swap(Bc[static_cast<size_t>(col + j * m)],
                  Bc[static_cast<size_t>(pivot + j * m)]);
      std::swap(rhsc[static_cast<size_t>(col)], rhsc[static_cast<size_t>(pivot)]);
    }
    for (int r = col + 1; r < m; r++) {
      double f = Bc[static_cast<size_t>(r + col * m)] /
                 Bc[static_cast<size_t>(col + col * m)];
      for (int j = col; j < m; j++)
        Bc[static_cast<size_t>(r + j * m)] -=
            f * Bc[static_cast<size_t>(col + j * m)];
      rhsc[static_cast<size_t>(r)] -= f * rhsc[static_cast<size_t>(col)];
    }
  }
  if (!ok) return;

  std::vector<double> coeffs(static_cast<size_t>(m), 0.0);
  for (int i = m - 1; i >= 0; i--) {
    double s = rhsc[static_cast<size_t>(i)];
    for (int j = i + 1; j < m; j++)
      s -= Bc[static_cast<size_t>(i + j * m)] * coeffs[static_cast<size_t>(j)];
    coeffs[static_cast<size_t>(i)] = s / Bc[static_cast<size_t>(i + i * m)];
  }

  // F_diis = F_hist * c[0:nvec]   (n2 × 1)
  CUDA_CHECK(cudaMemcpy(d_coeffs_, coeffs.data(),
                        static_cast<size_t>(nvec) * sizeof(double),
                        cudaMemcpyHostToDevice));
  cublasDgemv(blas, CUBLAS_OP_N, n2, nvec, &one, d_F_, n2, d_coeffs_, 1, &zero,
              d_Fock, 1);
}

}  // namespace cuest
