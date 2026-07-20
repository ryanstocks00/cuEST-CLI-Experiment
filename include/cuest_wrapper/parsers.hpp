#pragma once
/**
 * @file parsers.hpp
 * @brief GBS basis set, ECP, and XYZ file parsers (C++ reimplementations
 *        of the NVIDIA cuEST helper parsers).
 *
 * These parsers read the Gaussian94 Basis Set format files from EMSL Basis Set
 * Exchange (using "Gaussian" or "Psi4" format). Comment lines begin with '!'.
 */

#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "molecule.hpp"

namespace cuest {

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------
namespace detail {

inline std::string to_upper(const std::string& s) {
  std::string r = s;
  for (auto& c : r) c = static_cast<char>(toupper(static_cast<unsigned char>(c)));
  return r;
}

inline double parse_float_d(const std::string& s) {
  std::string buf = s;
  for (auto& c : buf)
    if (c == 'd' || c == 'D') c = 'e';
  return std::stod(buf);
}

inline int angular_momentum_to_l(char am) {
  static const char shell_types[] = {
      'S', 'P', 'D', 'F', 'G', 'H', 'I', 'K',
      'L', 'M', 'N', 'O', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};
  am = static_cast<char>(toupper(static_cast<unsigned char>(am)));
  for (int i = 0; i < 21; i++) {
    if (shell_types[i] == am) return i;
  }
  throw std::runtime_error(std::string("Unknown angular momentum symbol: ") + am);
}

inline int count_words(const std::string& s) {
  int n = 0;
  bool in_word = false;
  for (char c : s) {
    if (std::isspace(static_cast<unsigned char>(c))) {
      in_word = false;
    } else if (!in_word) {
      in_word = true;
      n++;
    }
  }
  return n;
}

inline std::string trim(const std::string& s) {
  size_t start = 0;
  while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])))
    start++;
  size_t end = s.size();
  while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1])))
    end--;
  return s.substr(start, end - start);
}

inline bool starts_with(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Parsed basis set per element
// ---------------------------------------------------------------------------
struct AtomBasisSet {
  size_t n_shells{0};
  std::vector<uint64_t> shell_types;    // L per shell
  std::vector<uint64_t> num_primitives;
  std::vector<uint64_t> primitive_offsets;
  std::vector<double> exponents;
  std::vector<double> coefficients;
};

inline AtomBasisSet parse_gbs_for_element(const std::string& filepath,
                                          const std::string& element) {
  std::ifstream fin(filepath);
  if (!fin) throw std::runtime_error("Cannot open GBS file: " + filepath);

  std::string elem_upper = detail::to_upper(element);

  // --- First pass: count shells and primitives ---
  size_t n_shells = 0, n_prims_total = 0;
  bool in_block = false, found_block = false;
  int nskip = 0;
  std::string line;

  while (std::getline(fin, line)) {
    std::string trimmed = detail::trim(line);
    if (trimmed.empty() || trimmed[0] == '!') continue;

    if (nskip > 0) { nskip--; continue; }

    if (trimmed.find("****") != std::string::npos) {
      if (found_block) break;
      in_block = false;
      continue;
    }

    int nwords = detail::count_words(trimmed);
    if (nwords != 2 && nwords != 3) continue;

    if (!in_block) {
      std::istringstream iss(trimmed);
      std::string block_atom;
      iss >> block_atom;
      block_atom = detail::to_upper(block_atom);
      if (block_atom == elem_upper) found_block = true;
      in_block = true;
      continue;
    }

    if (in_block && !found_block) continue;

    if (found_block) {
      std::istringstream iss(trimmed);
      std::string sh_type;
      int nprim;
      if (iss >> sh_type >> nprim) {
        nskip = nprim;
        n_shells++;
        n_prims_total += nprim;
      }
    }
  }

  if (!found_block) {
    throw std::runtime_error("Element " + element + " not found in " + filepath);
  }

  // --- Second pass: populate arrays ---
  fin.clear();
  fin.seekg(0);

  AtomBasisSet result;
  result.n_shells = n_shells;
  result.shell_types.resize(n_shells);
  result.num_primitives.resize(n_shells);
  result.primitive_offsets.resize(n_shells);
  result.exponents.resize(n_prims_total);
  result.coefficients.resize(n_prims_total);

  size_t nshell_idx = 0, nprim_idx = 0;
  in_block = false;
  found_block = false;
  int nprims_to_parse = 0;

  while (std::getline(fin, line)) {
    std::string trimmed = detail::trim(line);
    if (trimmed.empty() || trimmed[0] == '!') continue;

    if (nprims_to_parse > 0 && found_block) {
      std::istringstream iss(trimmed);
      std::string s1, s2;
      if (iss >> s1 >> s2) {
        result.exponents[nprim_idx] = detail::parse_float_d(s1);
        result.coefficients[nprim_idx] = detail::parse_float_d(s2);
        nprim_idx++;
        nprims_to_parse--;
        continue;
      }
    }

    if (trimmed.find("****") != std::string::npos) {
      if (found_block) break;
      in_block = false;
      continue;
    }

    int nwords = detail::count_words(trimmed);
    if (nwords != 2 && nwords != 3) continue;

    if (!in_block) {
      std::istringstream iss(trimmed);
      std::string block_atom;
      iss >> block_atom;
      if (detail::to_upper(block_atom) == elem_upper) found_block = true;
      in_block = true;
      continue;
    }
    if (in_block && !found_block) continue;

    if (found_block) {
      std::istringstream iss(trimmed);
      std::string sh_type;
      int nprim;
      if (iss >> sh_type >> nprim) {
        uint64_t L = detail::angular_momentum_to_l(sh_type[0]);
        result.shell_types[nshell_idx] = L;
        result.num_primitives[nshell_idx] = nprim;
        result.primitive_offsets[nshell_idx] = nprim_idx;
        nshell_idx++;
        nprims_to_parse = nprim;
      }
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Parsed ECP per element
// ---------------------------------------------------------------------------
struct ECPShellSet {
  size_t n_shells{0};
  size_t n_elec{0};
  std::vector<uint64_t> shell_types;
  std::vector<uint64_t> num_primitives;
  std::vector<uint64_t> primitive_offsets;
  std::vector<uint64_t> Ns;  // atomic number powers
  std::vector<double> exponents;
  std::vector<double> coefficients;
};

inline ECPShellSet parse_ecp_for_element(const std::string& filepath,
                                          const std::string& element) {
  std::ifstream fin(filepath);
  if (!fin) throw std::runtime_error("Cannot open ECP file: " + filepath);

  std::string elem_target =
      detail::to_upper(element) + "-ECP";

  // --- First pass: count ---
  size_t n_shells = 0, n_prims_total = 0;
  bool in_target_block = false;
  int nelec = 0, max_l = 0;
  int nskip = 0;
  std::string line;

  while (std::getline(fin, line)) {
    std::string trimmed = detail::trim(line);
    if (trimmed.empty() || trimmed[0] == '!') continue;

    if (nskip > 0) { nskip--; continue; }

    // Check for end-of-block separator
    if (trimmed.find("****") != std::string::npos) {
      if (in_target_block) break;
      continue;
    }

    int nwords = detail::count_words(trimmed);

    // ECP header (3 words): element, max_L, n_elec
    if (!in_target_block && nwords == 3) {
      std::istringstream iss(trimmed);
      std::string block_atom;
      iss >> block_atom;
      if (detail::to_upper(block_atom) == elem_target) {
        in_target_block = true;
        iss.clear(); iss.seekg(0);
        iss >> block_atom >> max_l >> nelec;
      }
      continue;
    }

    // Potential label (2 words) e.g. "s-ul potential"
    if (in_target_block && nwords == 2) {
      std::istringstream iss(trimmed);
      std::string shell_type, label;
      iss >> shell_type >> label;
      if (label.find("0") != std::string::npos) break;  // next block
      continue;
    }

    // Number of primitives (1 word)
    if (in_target_block && nwords == 1) {
      int nprim = std::stoi(trimmed);
      nskip = nprim;
      n_shells++;
      n_prims_total += nprim;
    }
  }

  if (!in_target_block) {
    return {};  // no ECP for this element
  }

  // --- Second pass: populate ---
  fin.clear();
  fin.seekg(0);

  ECPShellSet result;
  result.n_shells = n_shells;
  result.n_elec = nelec;
  result.shell_types.resize(n_shells);
  result.num_primitives.resize(n_shells);
  result.primitive_offsets.resize(n_shells);
  result.Ns.resize(n_prims_total);
  result.exponents.resize(n_prims_total);
  result.coefficients.resize(n_prims_total);

  size_t nshell_idx = 0, nprim_idx = 0;
  in_target_block = false;
  int nprims_to_parse = 0;

  while (std::getline(fin, line)) {
    std::string trimmed = detail::trim(line);
    if (trimmed.empty() || trimmed[0] == '!') continue;

    // Parse primitive data
    if (nprims_to_parse > 0 && in_target_block) {
      std::istringstream iss(trimmed);
      std::string s1, s2, s3;
      if (iss >> s1 >> s2 >> s3) {
        result.Ns[nprim_idx] = std::stoul(s1);
        result.exponents[nprim_idx] = detail::parse_float_d(s2);
        result.coefficients[nprim_idx] = detail::parse_float_d(s3);
        nprim_idx++;
        nprims_to_parse--;
        continue;
      }
    }

    int nwords = detail::count_words(trimmed);

    // Header line
    if (!in_target_block && nwords == 3) {
      std::istringstream iss(trimmed);
      std::string block_atom;
      iss >> block_atom;
      if (detail::to_upper(block_atom) == elem_target) {
        in_target_block = true;
      }
      continue;
    }

    // Potential label
    if (in_target_block && nwords == 2) {
      std::istringstream iss(trimmed);
      std::string shell_type, label;
      iss >> shell_type >> label;
      if (label.find("0") != std::string::npos) break;
      uint64_t L = detail::angular_momentum_to_l(shell_type[0]);
      result.shell_types[nshell_idx] = L;
      continue;
    }

    // Number of primitives
    if (in_target_block && nwords == 1) {
      int nprim = std::stoi(trimmed);
      result.num_primitives[nshell_idx] = nprim;
      result.primitive_offsets[nshell_idx] = nprim_idx;
      nshell_idx++;
      nprims_to_parse = nprim;
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// XYZ file parser
// ---------------------------------------------------------------------------
struct XYZData {
  size_t n_atoms{0};
  std::vector<std::string> symbols;
  std::vector<double> xyz;   // coordinates in bohr (3*n_atoms)
  std::vector<double> charges; // nuclear charges (negative for cuEST conv)
};

inline XYZData parse_xyz(const std::string& filepath,
                          double ang_to_bohr = 1.0 / 0.529177210903) {
  std::ifstream fin(filepath);
  if (!fin) throw std::runtime_error("Cannot open XYZ file: " + filepath);

  std::string line;
  if (!std::getline(fin, line))
    throw std::runtime_error("Failed to read atom count from XYZ");

  size_t natom = std::stoul(detail::trim(line));
  if (!std::getline(fin, line))
    throw std::runtime_error("Failed to read comment line from XYZ");

  XYZData data;
  data.n_atoms = natom;
  data.symbols.resize(natom);
  data.xyz.resize(3 * natom);
  data.charges.resize(natom);

  for (size_t i = 0; i < natom; i++) {
    if (!std::getline(fin, line))
      throw std::runtime_error("Unexpected end of XYZ file");

    std::istringstream iss(line);
    std::string sym;
    double x, y, z;
    if (!(iss >> sym >> x >> y >> z))
      throw std::runtime_error("Failed to parse atom " + std::to_string(i));

    data.symbols[i] = detail::to_upper(sym);
    data.xyz[3 * i] = x * ang_to_bohr;
    data.xyz[3 * i + 1] = y * ang_to_bohr;
    data.xyz[3 * i + 2] = z * ang_to_bohr;

    // Determine atomic number using shared element lookup
    int atomic_z = Molecule::symbol_to_z(data.symbols[i]);
    data.charges[i] = -1.0 * static_cast<double>(atomic_z);
  }

  return data;
}

}  // namespace cuest
