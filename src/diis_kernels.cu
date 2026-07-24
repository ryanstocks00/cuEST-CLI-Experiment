/**
 * @file diis_kernels.cu
 * @brief Pack/unpack the upper triangle of an N×N matrix for DIIS history.
 *
 * DIIS stores a Fock matrix and an error matrix per history vector per spin
 * channel. Both are triangular-redundant — the Fock is symmetric and the
 * FDS−SDF error is antisymmetric — so keeping only the upper triangle halves
 * the history, which is what pays for extrapolating both spin channels jointly
 * at unchanged memory.
 *
 * The same packing serves both: for the antisymmetric error the diagonal is
 * zero and the lower triangle is redundant, so packing the upper triangle
 * (including that zero diagonal) discards nothing. It does halve the Gram
 * matrix, but uniformly — and the Pulay coefficients are invariant under a
 * uniform scaling of B (see DIIS::extrapolate), so the extrapolation is
 * unchanged.
 */
#include <cuda_runtime.h>

namespace cuest {
namespace {

/// Row-major index of (i, j), j >= i, in the packed upper triangle.
__device__ __forceinline__ long packed_index(int i, int j, int N) {
  // Rows 0..i-1 contribute N, N-1, ... N-i+1 entries.
  const long base = static_cast<long>(i) * N - (static_cast<long>(i) * (i - 1)) / 2;
  return base + (j - i);
}

__global__ void pack_upper_kernel(const double* __restrict__ A,
                                  double* __restrict__ out, int N) {
  const long total = (static_cast<long>(N) * (N + 1)) / 2;
  for (long t = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
       t < total; t += static_cast<long>(gridDim.x) * blockDim.x) {
    // Invert the triangular index: find the row i containing offset t.
    // i = floor((2N+1 - sqrt((2N+1)^2 - 8t)) / 2), computed in double then
    // corrected, which is exact for any N that fits in device memory.
    const double b = 2.0 * N + 1.0;
    int i = static_cast<int>((b - sqrt(b * b - 8.0 * static_cast<double>(t))) * 0.5);
    while (i > 0 && packed_index(i, i, N) > t) --i;
    while (packed_index(i + 1, i + 1, N) <= t) ++i;
    const int j = static_cast<int>(t - packed_index(i, i, N)) + i;
    out[t] = A[static_cast<long>(i) * N + j];
  }
}

__global__ void unpack_upper_kernel(const double* __restrict__ in,
                                    double* __restrict__ A, int N, bool symmetric) {
  const long total = (static_cast<long>(N) * (N + 1)) / 2;
  for (long t = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
       t < total; t += static_cast<long>(gridDim.x) * blockDim.x) {
    const double b = 2.0 * N + 1.0;
    int i = static_cast<int>((b - sqrt(b * b - 8.0 * static_cast<double>(t))) * 0.5);
    while (i > 0 && packed_index(i, i, N) > t) --i;
    while (packed_index(i + 1, i + 1, N) <= t) ++i;
    const int j = static_cast<int>(t - packed_index(i, i, N)) + i;
    const double v = in[t];
    A[static_cast<long>(i) * N + j] = v;
    if (i != j) A[static_cast<long>(j) * N + i] = symmetric ? v : -v;
  }
}

int blocks_for(long total) {
  const long b = (total + 255) / 256;
  return static_cast<int>(b > 65535 ? 65535 : (b < 1 ? 1 : b));
}

}  // namespace

void diis_pack_upper(const double* d_A, double* d_out, int N, cudaStream_t stream) {
  const long total = (static_cast<long>(N) * (N + 1)) / 2;
  pack_upper_kernel<<<blocks_for(total), 256, 0, stream>>>(d_A, d_out, N);
}

void diis_unpack_upper(const double* d_in, double* d_A, int N, bool symmetric,
                       cudaStream_t stream) {
  const long total = (static_cast<long>(N) * (N + 1)) / 2;
  unpack_upper_kernel<<<blocks_for(total), 256, 0, stream>>>(d_in, d_A, N, symmetric);
}

}  // namespace cuest
