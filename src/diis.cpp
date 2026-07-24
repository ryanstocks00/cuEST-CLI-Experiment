/**
 * @file diis.cpp — Device Pulay DIIS (cuBLAS; history as n2×nvec matrices).
 */
#include "diis.hpp"

#include "cuest_wrapper/nvtx.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

namespace cuest {

void DIIS::init(int max_space, int N, int nchan) {
  N_ = N;
  n2_ = N * N;
  packed_ = N * (N + 1) / 2;
  nchan_ = nchan;
  vec_ = packed_ * nchan_;
  max_space_ = max_space;
  count_ = 0;
  residual_ready_ = false;
  last_rms_ = 0.0;

  const size_t n2_bytes = static_cast<size_t>(n2_) * sizeof(double);
  const size_t vec_bytes = static_cast<size_t>(vec_) * sizeof(double);
  const size_t hist_bytes = vec_bytes * static_cast<size_t>(max_space);
  d_E_.alloc(hist_bytes);
  d_F_.alloc(hist_bytes);
  d_gram_.alloc(static_cast<size_t>(max_space) * max_space * sizeof(double));
  d_coeffs_.alloc(static_cast<size_t>(max_space) * sizeof(double));
  d_tmp1_.alloc(n2_bytes);
  d_tmp2_.alloc(n2_bytes);
  d_full_.alloc(n2_bytes);
  d_err_.alloc(vec_bytes);
  d_work_.alloc(vec_bytes);
}

void DIIS::clear() {
  count_ = 0;
  residual_ready_ = false;
}

void DIIS::push(cublasHandle_t blas, const double* d_err, const double* d_fock) {
  // Keep columns [0, count) packed oldest→newest for a single GEMM.
  if (count_ == max_space_) {
    for (int j = 0; j < max_space_ - 1; j++) {
      cublasDcopy(blas, vec_, err_col(j + 1), 1, err_col(j), 1);
      cublasDcopy(blas, vec_, fock_col(j + 1), 1, fock_col(j), 1);
    }
    count_ = max_space_ - 1;
  }
  cublasDcopy(blas, vec_, d_err, 1, err_col(count_), 1);
  cublasDcopy(blas, vec_, d_fock, 1, fock_col(count_), 1);
  ++count_;
}

void DIIS::residual_channel(cublasHandle_t blas, const double* d_S,
                            const double* d_Fock, const double* d_D,
                            double* d_out) {
  const int N = N_;
  const int n2 = n2_;
  double one = 1.0, zero = 0.0, minus_one = -1.0;

  // FDS = (F*D)*S → d_tmp2_
  CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one,
              d_Fock, N, d_D, N, &zero, d_tmp1_, N));
  CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one,
              d_tmp1_, N, d_S, N, &zero, d_tmp2_, N));
  // SDF = (S*D)*F → d_out
  CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one,
              d_S, N, d_D, N, &zero, d_tmp1_, N));
  CUBLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one,
              d_tmp1_, N, d_Fock, N, &zero, d_out, N));
  // d_out = FDS - SDF
  CUBLAS_CHECK(cublasDaxpy(blas, n2, &minus_one, d_out, 1, d_tmp2_, 1));
  CUBLAS_CHECK(cublasDcopy(blas, n2, d_tmp2_, 1, d_out, 1));
}

double DIIS::compute_residual(cublasHandle_t blas, const double* d_S,
                              const double* d_Fock_a, const double* d_D_a,
                              const double* d_Fock_b, const double* d_D_b) {
  // Both channels go into one residual vector, so the Pulay solve below sees a
  // single coupled error rather than two independent ones.
  residual_channel(blas, d_S, d_Fock_a, d_D_a, d_full_);
  diis_pack_upper(d_full_, d_err_.get(), N_);
  if (nchan_ == 2) {
    residual_channel(blas, d_S, d_Fock_b, d_D_b, d_full_);
    diis_pack_upper(d_full_, d_err_.get() + packed_, N_);
  }

  double ss = 0.0;
  CUBLAS_CHECK(cublasDdot(blas, vec_, d_err_, 1, d_err_, 1, &ss));
  // The error is antisymmetric with a zero diagonal, so the packed triangle
  // carries exactly half of the full matrix's squared norm.
  last_rms_ = std::sqrt(2.0 * ss / static_cast<double>(n2_ * nchan_));
  residual_ready_ = true;
  return last_rms_;
}

void DIIS::extrapolate(cublasHandle_t blas, DeviceArray<double>& d_Fock_a,
                       DeviceArray<double>* d_Fock_b) {
  NvtxRange range("DIIS");
  if (!residual_ready_) return;

  double one = 1.0, zero = 0.0;

  // Pack both Focks into one history vector, matching the residual layout.
  diis_pack_upper(d_Fock_a, d_work_.get(), N_);
  if (nchan_ == 2) diis_pack_upper(*d_Fock_b, d_work_.get() + packed_, N_);
  push(blas, d_err_, d_work_);
  residual_ready_ = false;

  const int nvec = count_;
  if (nvec < 2) return;

  // Gram = Eᵀ E  (nvec × nvec), E is vec_ × nvec column-major
  cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, nvec, nvec, vec_, &one,
              d_E_, vec_, d_E_, vec_, &zero, d_gram_, nvec);

  std::vector<double> gram(static_cast<size_t>(nvec) * nvec);
  CUDA_CHECK(cudaMemcpy(gram.data(), d_gram_,
                        gram.size() * sizeof(double), cudaMemcpyDeviceToHost));

  // Jacobi (diagonal) preconditioning: c_i = dscale[i] * c'_i, so the scaled
  // Gram block dscale[i]*Gram[i,j]*dscale[j] has unit diagonal.
  std::vector<double> dscale(static_cast<size_t>(nvec));
  for (int i = 0; i < nvec; i++) {
    double d = gram[static_cast<size_t>(i + i * nvec)];
    dscale[static_cast<size_t>(i)] = (d > 1e-300) ? 1.0 / std::sqrt(d) : 1.0;
  }

  // Augmented Pulay system: [Gram'  -d; -dᵀ  0] c' = [0; -1]
  const int m = nvec + 1;
  std::vector<double> B(static_cast<size_t>(m) * m, 0.0);
  for (int j = 0; j < nvec; j++)
    for (int i = 0; i < nvec; i++)
      B[static_cast<size_t>(i + j * m)] = dscale[static_cast<size_t>(i)] *
          gram[static_cast<size_t>(i + j * nvec)] * dscale[static_cast<size_t>(j)];
  for (int i = 0; i < nvec; i++) {
    B[static_cast<size_t>(i + nvec * m)] = -dscale[static_cast<size_t>(i)];
    B[static_cast<size_t>(nvec + i * m)] = -dscale[static_cast<size_t>(i)];
  }
  std::vector<double> rhs(static_cast<size_t>(m), 0.0);
  rhs[static_cast<size_t>(nvec)] = -1.0;

  const double pivot_tol = 1e-10;

  auto Bc = B;
  auto rhsc = rhs;
  bool ok = true;
  for (int col = 0; col < m; col++) {
    int pivot = col;
    for (int r = col + 1; r < m; r++)
      if (std::fabs(Bc[static_cast<size_t>(r + col * m)]) >
          std::fabs(Bc[static_cast<size_t>(pivot + col * m)]))
        pivot = r;
    if (std::fabs(Bc[static_cast<size_t>(pivot + col * m)]) < pivot_tol) {
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
  if (!ok) {
    // Singular / linearly dependent history: drop everything and keep the
    // raw (pre-extrapolation) Fock that the caller still holds in d_Fock.
    clear();
    return;
  }

  std::vector<double> coeffs(static_cast<size_t>(m), 0.0);
  for (int i = m - 1; i >= 0; i--) {
    double s = rhsc[static_cast<size_t>(i)];
    for (int j = i + 1; j < m; j++)
      s -= Bc[static_cast<size_t>(i + j * m)] * coeffs[static_cast<size_t>(j)];
    coeffs[static_cast<size_t>(i)] = s / Bc[static_cast<size_t>(i + i * m)];
  }
  // Undo the Jacobi scaling: actual c_i = dscale[i] * c'_i.
  for (int i = 0; i < nvec; i++)
    coeffs[static_cast<size_t>(i)] *= dscale[static_cast<size_t>(i)];

  // F_diis = F_hist * c[0:nvec], then unpack back into each channel. One
  // coefficient set drives both spins — that is the whole point.
  CUDA_CHECK(cudaMemcpy(d_coeffs_, coeffs.data(),
                        static_cast<size_t>(nvec) * sizeof(double),
                        cudaMemcpyHostToDevice));
  cublasDgemv(blas, CUBLAS_OP_N, vec_, nvec, &one, d_F_, vec_, d_coeffs_, 1,
              &zero, d_work_, 1);
  diis_unpack_upper(d_work_.get(), d_Fock_a, N_, /*symmetric=*/true);
  if (nchan_ == 2)
    diis_unpack_upper(d_work_.get() + packed_, *d_Fock_b, N_, /*symmetric=*/true);
}

}  // namespace cuest
