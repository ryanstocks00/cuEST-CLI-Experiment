#pragma once
/**
 * @file diis.hpp
 * @brief Device-side Pulay DIIS (FDS − SDF error; history as n2×nvec matrices).
 */

#include <cublas_v2.h>

#include "cuest_wrapper/raii.hpp"

namespace cuest {

/// Pulay DIIS: error / Fock history stored as column-major (n2 × max_space)
/// matrices so Gram(B) = EᵀE and F_diis = F·c are single BLAS calls.
class DIIS {
 public:
  DIIS() = default;

  /// Allocate history matrices + GEMM scratch for AO matrices of size N×N.
  void init(int max_space, int N);

  /// Drop history (keeps allocated buffers).
  void clear();

  [[nodiscard]] int size() const { return count_; }

  /// Build error = FDS − SDF, push into the subspace, and overwrite d_Fock
  /// with the DIIS-extrapolated Fock when the subspace has ≥ 2 vectors.
  void extrapolate(cublasHandle_t blas, const double* d_S,
                   DeviceArray<double>& d_Fock, const double* d_D);

 private:
  void push(cublasHandle_t blas, const double* d_err, const double* d_fock);
  double* err_col(int j) { return d_E_.get() + static_cast<size_t>(j) * n2_; }
  double* fock_col(int j) { return d_F_.get() + static_cast<size_t>(j) * n2_; }

  int N_{0};
  int n2_{0};
  int max_space_{0};
  int count_{0};

  DeviceArray<double> d_E_;       // n2 × max_space  (error columns)
  DeviceArray<double> d_F_;       // n2 × max_space  (Fock columns)
  DeviceArray<double> d_gram_;    // max_space × max_space  (EᵀE)
  DeviceArray<double> d_coeffs_;  // max_space
  DeviceArray<double> d_tmp1_;
  DeviceArray<double> d_tmp2_;
  DeviceArray<double> d_err_;
};

}  // namespace cuest
