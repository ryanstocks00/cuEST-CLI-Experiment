#pragma once
/**
 * @file molecule.hpp
 * @brief Molecule representation: atoms, coordinates, charges.
 */

#include <cmath>
#include <string>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "constants.hpp"

namespace cuest {

// ---------------------------------------------------------------------------
// Atom
// ---------------------------------------------------------------------------
struct Atom {
  std::string symbol;
  double x{0.0}, y{0.0}, z{0.0};  // coordinates in bohr
  int atomic_number{0};
};

// ---------------------------------------------------------------------------
// Molecule - holds atom list and coordinates in both host and device memory
// ---------------------------------------------------------------------------
class Molecule {
 public:
  Molecule() = default;

  // Add atom with coordinates in angstrom (converts to bohr internally)
  void add_atom(const std::string& symbol, double x_ang, double y_ang,
                double z_ang,
                double ang_to_bohr = constants::angstrom_per_bohr) {
    Atom atom;
    atom.symbol = symbol;
    atom.x = x_ang * ang_to_bohr;
    atom.y = y_ang * ang_to_bohr;
    atom.z = z_ang * ang_to_bohr;
    atom.atomic_number = symbol_to_z(symbol);
    atoms_.push_back(atom);
  }

  // Add atom with coordinates already in bohr (no conversion)
  void add_atom_bohr(const std::string& symbol, double x_bohr,
                     double y_bohr, double z_bohr) {
    Atom atom;
    atom.symbol = symbol;
    atom.x = x_bohr;
    atom.y = y_bohr;
    atom.z = z_bohr;
    atom.atomic_number = symbol_to_z(symbol);
    atoms_.push_back(atom);
  }

  [[nodiscard]] size_t natom() const { return atoms_.size(); }
  [[nodiscard]] const Atom& atom(size_t i) const {
    if (i >= atoms_.size())
      throw std::out_of_range("Atom index " + std::to_string(i) +
                              " out of range [0, " + std::to_string(atoms_.size()) + ")");
    return atoms_[i];
  }
  [[nodiscard]] const std::vector<Atom>& atoms() const { return atoms_; }

  // Total number of electrons (neutral)
  [[nodiscard]] int nelec() const {
    int nel = 0;
    for (const auto& a : atoms_) nel += a.atomic_number;
    return nel;
  }

  // Number of occupied orbitals (RHF/RKS: nalpha = nbeta = nelec/2)
  [[nodiscard]] int nocc() const { return nelec() / 2; }

  // Spin multiplicity info
  [[nodiscard]] int multiplicity() const { return multiplicity_; }
  void set_multiplicity(int mult) { multiplicity_ = mult; }
  [[nodiscard]] int nalpha() const { return (nelec() + multiplicity_ - 1) / 2; }
  [[nodiscard]] int nbeta() const { return (nelec() - multiplicity_ + 1) / 2; }

  // Coordinate arrays for cuEST
  [[nodiscard]] std::vector<double> xyz_host() const {
    std::vector<double> xyz(natom() * 3);
    for (size_t i = 0; i < natom(); i++) {
      xyz[3 * i] = atoms_[i].x;
      xyz[3 * i + 1] = atoms_[i].y;
      xyz[3 * i + 2] = atoms_[i].z;
    }
    return xyz;
  }

  // Charges for cuEST: -1 * nuclear_charge (electron = -1 convention)
  [[nodiscard]] std::vector<double> charges_host() const {
    std::vector<double> c(natom());
    for (size_t i = 0; i < natom(); i++)
      c[i] = -1.0 * atoms_[i].atomic_number;
    return c;
  }

  // Unique element symbols (O(1) via unordered_set)
  [[nodiscard]] std::vector<std::string> unique_symbols() const {
    std::unordered_set<std::string> seen;
    std::vector<std::string> uniq;
    for (const auto& a : atoms_) {
      if (seen.insert(a.symbol).second)
        uniq.push_back(a.symbol);
    }
    return uniq;
  }

  // Total nuclear repulsion energy
  [[nodiscard]] double nuclear_repulsion() const {
    double enuc = 0.0;
    for (size_t i = 0; i < natom(); i++) {
      for (size_t j = i + 1; j < natom(); j++) {
        double dx = atoms_[i].x - atoms_[j].x;
        double dy = atoms_[i].y - atoms_[j].y;
        double dz = atoms_[i].z - atoms_[j].z;
        double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        enuc += atoms_[i].atomic_number * atoms_[j].atomic_number / r;
      }
    }
    return enuc;
  }

  // Symbol-to-atomic-number lookup (shared with parsers)
  [[nodiscard]] static int symbol_to_z(const std::string& sym) {
    static const char* elements[] = {
        "X",  "H",  "HE", "LI", "BE", "B",  "C",  "N",  "O",  "F",  "NE",
        "NA", "MG", "AL", "SI", "P",  "S",  "CL", "AR", "K",  "CA",
        "SC", "TI", "V",  "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
        "GA", "GE", "AS", "SE", "BR", "KR", "RB", "SR", "Y",  "ZR",
        "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN",
        "SB", "TE", "I",  "XE", "CS", "BA", "LA", "CE", "PR", "ND",
        "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB",
        "LU", "HF", "TA", "W",  "RE", "OS", "IR", "PT", "AU", "HG",
        "TL", "PB", "BI", "PO", "AT", "RN", "FR", "RA", "AC", "TH",
        "PA", "U",  "NP", "PU", "AM", "CM", "BK", "CF", "ES", "FM",
        "MD", "NO", "LR", "RF", "DB", "SG", "BH", "HS", "MT", "DS",
        "RG", "CN", "NH", "FL", "MC", "LV", "TS", "OG"};
    std::string upper = sym;
    for (auto& c : upper) c = static_cast<char>(toupper(c));
    for (int i = 0; i < 119; i++) {
      if (upper == elements[i]) return i;
    }
    throw std::runtime_error("Unknown element: " + sym);
  }

 private:

  std::vector<Atom> atoms_;
  int multiplicity_{1};  // 1=singlet, 2=doublet, 3=triplet, etc.
};

}  // namespace cuest
