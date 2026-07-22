#pragma once
/**
 * @file diis.hpp
 * @brief Device-side Pulay DIIS (FDS − SDF error), packed-triangle history.
 */

#include <cublas_v2.h>

#include "cuest_wrapper/raii.hpp"

namespace cuest {

/// Pack / unpack the upper triangle of an N×N matrix (see diis_kernels.cu).
void diis_pack_upper(const double* d_A, double* d_out, int N,
                     cudaStream_t stream = nullptr);
void diis_unpack_upper(const double* d_in, double* d_A, int N, bool symmetric,
                       cudaStream_t stream = nullptr);

/// Pulay DIIS over one or both spin channels.
///
/// ## One coefficient set for both spins
///
/// For UKS the α and β Fock matrices are *coupled* — each depends on the other
/// through J (and through Vxc) — so they are not two independent minimisation
/// problems. Extrapolating them separately, with a coefficient set fitted to
/// each channel's own error, lets the two channels pull in different directions
/// and converges poorly for open-shell systems. PySCF instead treats the pair
/// as a single object: one error vector formed by concatenating both channels,
/// one Pulay solve, one set of coefficients applied to both. This class does
/// the same — `nchan` is 2 for UKS and 1 for RKS.
///
/// ## Packed history
///
/// Each history entry keeps only the upper triangle of each matrix, halving the
/// storage and so paying for the second channel. The Fock is symmetric and the
/// FDS−SDF error is antisymmetric, so in both cases the lower triangle is
/// redundant; unpacking restores the correct sign. Halving the error vector
/// halves the Gram matrix uniformly, and the Pulay coefficients are invariant
/// under a uniform scaling of B, so the extrapolation is unaffected.
///
/// History is stored column-major (packed·nchan) × max_space, so Gram = EᵀE and
/// F_diis = F·c remain single BLAS calls.
class DIIS {
 public:
  DIIS() = default;

  /// Allocate history + scratch for `nchan` spin channels of N×N matrices.
  void init(int max_space, int N, int nchan = 1);

  /// Drop history (keeps allocated buffers).
  void clear();

  [[nodiscard]] int size() const { return count_; }
  [[nodiscard]] int channels() const { return nchan_; }

  /// Build error = FDS − SDF for every channel into the internal residual
  /// buffer and return the RMS over the full (unpacked) matrices. Call once per
  /// iteration; extrapolate() reuses that residual. `d_Fock_b`/`d_D_b` are
  /// required when nchan == 2 and ignored otherwise.
  double compute_residual(cublasHandle_t blas, const double* d_S,
                          const double* d_Fock_a, const double* d_D_a,
                          const double* d_Fock_b = nullptr,
                          const double* d_D_b = nullptr);

  /// Push the residual from the last compute_residual() together with the Fock
  /// matrices, then overwrite them with the DIIS-extrapolated Focks once the
  /// subspace has ≥ 2 vectors. On a singular Pulay solve, clears history and
  /// leaves the Focks unchanged. `d_Fock_b` is required when nchan == 2.
  void extrapolate(cublasHandle_t blas, DeviceArray<double>& d_Fock_a,
                   DeviceArray<double>* d_Fock_b = nullptr);

  /// RMS of the residual from the last compute_residual() (0 if none yet).
  [[nodiscard]] double last_error_rms() const { return last_rms_; }

 private:
  void push(cublasHandle_t blas, const double* d_err, const double* d_fock);
  /// Build FDS − SDF for one channel into `d_out` (full N×N).
  void residual_channel(cublasHandle_t blas, const double* d_S,
                        const double* d_Fock, const double* d_D, double* d_out);
  double* err_col(int j) { return d_E_.get() + static_cast<size_t>(j) * vec_; }
  double* fock_col(int j) { return d_F_.get() + static_cast<size_t>(j) * vec_; }

  int N_{0};
  int n2_{0};
  int packed_{0};   // N(N+1)/2, one channel
  int nchan_{1};
  int vec_{0};      // packed_ * nchan_, one history vector
  int max_space_{0};
  int count_{0};
  bool residual_ready_{false};
  double last_rms_{0.0};

  DeviceArray<double> d_E_;       // vec_ × max_space  (error columns, packed)
  DeviceArray<double> d_F_;       // vec_ × max_space  (Fock columns, packed)
  DeviceArray<double> d_gram_;    // max_space × max_space  (EᵀE)
  DeviceArray<double> d_coeffs_;  // max_space
  DeviceArray<double> d_tmp1_;
  DeviceArray<double> d_tmp2_;
  DeviceArray<double> d_full_;    // one unpacked N×N scratch
  DeviceArray<double> d_err_;     // vec_  (packed residual, all channels)
  DeviceArray<double> d_work_;    // vec_  (packed extrapolation result)
};

}  // namespace cuest
