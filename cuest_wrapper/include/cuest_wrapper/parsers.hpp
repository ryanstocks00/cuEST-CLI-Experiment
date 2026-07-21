#pragma once
/**
 * @file parsers.hpp
 * @brief XYZ geometry file parser.
 */

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "constants.hpp"
#include "molecule.hpp"

namespace cuest {

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------
namespace detail {

inline std::string to_upper(const std::string& s) {
  std::string r = s;
  for (auto& c : r) c = static_cast<char>(toupper(static_cast<unsigned char>(c)));
  return r;
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

}  // namespace detail

// ---------------------------------------------------------------------------
// XYZ data
// ---------------------------------------------------------------------------
struct XYZData {
  size_t n_atoms{0};
  std::vector<std::string> symbols;
  std::vector<double> xyz;      // coordinates in bohr (3 * n_atoms)
  std::vector<double> charges;  // nuclear charges (negative for cuEST conv)
};

// ---------------------------------------------------------------------------
// XYZ file parser
// ---------------------------------------------------------------------------
inline XYZData parse_xyz(const std::string& filepath,
                          double ang_to_bohr = constants::bohr_per_angstrom) {
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
    data.xyz[3 * i]     = x * ang_to_bohr;
    data.xyz[3 * i + 1] = y * ang_to_bohr;
    data.xyz[3 * i + 2] = z * ang_to_bohr;

    int atomic_z = Molecule::symbol_to_z(data.symbols[i]);
    data.charges[i] = -1.0 * static_cast<double>(atomic_z);
  }

  return data;
}

}  // namespace cuest
