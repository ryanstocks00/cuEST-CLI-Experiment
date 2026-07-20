#pragma once
/**
 * @file shell_norm.hpp
 * @brief Shell coefficient normalization (C++ port of NVIDIA cuEST sample helper).
 */

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace cuest {

inline std::vector<double> compute_normalized_coefficients(
    size_t L,
    size_t num_primitives,
    const double* exponents,
    const double* coefficients,
    double normalization = 1.0) {

  if (L >= 9) throw std::runtime_error("L >= 9 not supported for normalization");
  if (normalization <= 0.0)
    throw std::runtime_error("Normalization must be positive");

  for (size_t i = 0; i < num_primitives; i++)
    if (exponents[i] <= 0.0)
      throw std::runtime_error("Exponents must be positive");

  double pi32 = std::pow(M_PI, 1.5);
  double twoL = std::pow(2.0, static_cast<double>(L));

  // Double factorial for odd numbers: (2L-1)!!
  size_t dfact = 1;
  for (size_t l = 1; l <= L; l++) dfact *= 2 * l - 1;

  std::vector<double> coeff_norm(num_primitives);

  for (size_t i = 0; i < num_primitives; i++) {
    coeff_norm[i] = std::sqrt(twoL / (pi32 * static_cast<double>(dfact)) *
                               std::pow(2.0 * exponents[i],
                                        static_cast<double>(L) + 1.5)) *
                     coefficients[i];
  }

  // Q matrix normalization
  double Q = 0.0;
  for (size_t i = 0; i < num_primitives; i++) {
    for (size_t j = 0; j < num_primitives; j++) {
      Q += std::pow(std::sqrt(4.0 * exponents[i] * exponents[j]) /
                        (exponents[i] + exponents[j]),
                    static_cast<double>(L) + 1.5) *
            coeff_norm[i] * coeff_norm[j];
    }
  }
  if (Q <= 0.0 || !std::isfinite(Q))
    throw std::runtime_error("Degenerate contraction (Q <= 0) in shell normalization");
  Q = std::pow(Q, -0.5) * std::sqrt(normalization);

  for (size_t i = 0; i < num_primitives; i++) coeff_norm[i] *= Q;

  return coeff_norm;
}

}  // namespace cuest
