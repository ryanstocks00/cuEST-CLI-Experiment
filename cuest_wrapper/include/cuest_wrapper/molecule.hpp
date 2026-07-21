#pragma once
/**
 * @file molecule.hpp
 * @brief Molecule representation: atoms, coordinates, charge/multiplicity.
 *
 * Geometry only — ECP core counts belong with the basis, not the molecule.
 * Electron-count helpers take a total core-electron count; Z_eff / Enuc /
 * nuclear charges need the per-atom core vector from the ECP basis.
 */

#include <cctype>
#include <cmath>
#include <span>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "constants.hpp"

namespace cuest {

struct Atom {
  int atomic_number{0};
  double x{0.0}, y{0.0}, z{0.0};  // coordinates in bohr
};

class Molecule {
 public:
  Molecule() = default;

  void add_atom(int atomic_number, double x_ang, double y_ang, double z_ang,
                double ang_to_bohr = constants::bohr_per_angstrom) {
    Atom atom;
    atom.atomic_number = atomic_number;
    atom.x = x_ang * ang_to_bohr;
    atom.y = y_ang * ang_to_bohr;
    atom.z = z_ang * ang_to_bohr;
    atoms_.push_back(atom);
  }

  void add_atom(const std::string& symbol, double x_ang, double y_ang,
                double z_ang,
                double ang_to_bohr = constants::bohr_per_angstrom) {
    add_atom(symbol_to_z(symbol), x_ang, y_ang, z_ang, ang_to_bohr);
  }

  void add_atom_bohr(int atomic_number, double x_bohr, double y_bohr,
                     double z_bohr) {
    Atom atom;
    atom.atomic_number = atomic_number;
    atom.x = x_bohr;
    atom.y = y_bohr;
    atom.z = z_bohr;
    atoms_.push_back(atom);
  }

  void add_atom_bohr(const std::string& symbol, double x_bohr, double y_bohr,
                     double z_bohr) {
    add_atom_bohr(symbol_to_z(symbol), x_bohr, y_bohr, z_bohr);
  }

  [[nodiscard]] size_t natom() const { return atoms_.size(); }
  [[nodiscard]] const Atom& atom(size_t i) const {
    if (i >= atoms_.size())
      throw std::out_of_range("Atom index " + std::to_string(i) +
                              " out of range [0, " + std::to_string(atoms_.size()) + ")");
    return atoms_[i];
  }
  [[nodiscard]] const std::vector<Atom>& atoms() const { return atoms_; }

  void set_charge(int charge) { charge_ = charge; }
  [[nodiscard]] int charge() const { return charge_; }

  [[nodiscard]] int multiplicity() const { return multiplicity_; }
  void set_multiplicity(int mult) { multiplicity_ = mult; }

  /// Z_eff = Z - n_core (n_core from the ECP basis for that atom).
  [[nodiscard]] int zeff(size_t i, int n_core = 0) const {
    if (n_core < 0 || n_core > atoms_[i].atomic_number)
      throw std::runtime_error("Invalid ECP electron count for atom " +
                               std::to_string(i));
    return atoms_[i].atomic_number - n_core;
  }

  /// Valence electrons: sum(Z) - charge - n_ecp (total cores replaced by ECP).
  [[nodiscard]] int nelec(int n_ecp = 0) const {
    int nel = 0;
    for (const auto& a : atoms_) nel += a.atomic_number;
    return nel - charge_ - n_ecp;
  }

  [[nodiscard]] int nocc(int n_ecp = 0) const { return nelec(n_ecp) / 2; }
  [[nodiscard]] int nalpha(int n_ecp = 0) const {
    return (nelec(n_ecp) + multiplicity_ - 1) / 2;
  }
  [[nodiscard]] int nbeta(int n_ecp = 0) const {
    return (nelec(n_ecp) - multiplicity_ + 1) / 2;
  }

  [[nodiscard]] std::vector<double> xyz_host() const {
    std::vector<double> xyz(natom() * 3);
    for (size_t i = 0; i < natom(); i++) {
      xyz[3 * i] = atoms_[i].x;
      xyz[3 * i + 1] = atoms_[i].y;
      xyz[3 * i + 2] = atoms_[i].z;
    }
    return xyz;
  }

  /// Nuclear charges for cuEST: -Z_eff. Empty ecp_cores ⇒ all-electron.
  [[nodiscard]] std::vector<double> charges_host(
      std::span<const int> ecp_cores = {}) const {
    validate_ecp_cores(ecp_cores);
    std::vector<double> c(natom());
    for (size_t i = 0; i < natom(); i++)
      c[i] = -1.0 * static_cast<double>(
                 zeff(i, ecp_cores.empty() ? 0 : ecp_cores[i]));
    return c;
  }

  [[nodiscard]] double nuclear_repulsion(
      std::span<const int> ecp_cores = {}) const {
    validate_ecp_cores(ecp_cores);
    double enuc = 0.0;
    for (size_t i = 0; i < natom(); i++) {
      const int zi = zeff(i, ecp_cores.empty() ? 0 : ecp_cores[i]);
      for (size_t j = i + 1; j < natom(); j++) {
        const int zj = zeff(j, ecp_cores.empty() ? 0 : ecp_cores[j]);
        double dx = atoms_[i].x - atoms_[j].x;
        double dy = atoms_[i].y - atoms_[j].y;
        double dz = atoms_[i].z - atoms_[j].z;
        double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < 1e-20)
          throw std::runtime_error("Coincident atoms in nuclear repulsion");
        enuc += static_cast<double>(zi) * static_cast<double>(zj) /
                std::sqrt(r2);
      }
    }
    return enuc;
  }

  [[nodiscard]] static int symbol_to_z(const std::string& sym) {
    static const auto& table = element_table();
    std::string upper = sym;
    for (auto& c : upper)
      c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    auto it = table.find(upper);
    if (it == table.end())
      throw std::runtime_error("Unknown element: " + sym);
    return it->second;
  }

  [[nodiscard]] static const char* z_to_symbol(int z) {
    static const char* elements[] = {
        "X",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
        "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
        "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};
    if (z < 0 || z > 118)
      throw std::out_of_range("atomic number out of range");
    return elements[z];
  }

 private:
  void validate_ecp_cores(std::span<const int> ecp_cores) const {
    if (!ecp_cores.empty() && ecp_cores.size() != natom())
      throw std::runtime_error("ECP core vector size != natom");
  }

  [[nodiscard]] static const std::unordered_map<std::string, int>&
  element_table() {
    static const std::unordered_map<std::string, int> table = {
        {"X", 0},   {"H", 1},   {"HE", 2},  {"LI", 3},  {"BE", 4},
        {"B", 5},   {"C", 6},   {"N", 7},   {"O", 8},   {"F", 9},
        {"NE", 10}, {"NA", 11}, {"MG", 12}, {"AL", 13}, {"SI", 14},
        {"P", 15},  {"S", 16},  {"CL", 17}, {"AR", 18}, {"K", 19},
        {"CA", 20}, {"SC", 21}, {"TI", 22}, {"V", 23},  {"CR", 24},
        {"MN", 25}, {"FE", 26}, {"CO", 27}, {"NI", 28}, {"CU", 29},
        {"ZN", 30}, {"GA", 31}, {"GE", 32}, {"AS", 33}, {"SE", 34},
        {"BR", 35}, {"KR", 36}, {"RB", 37}, {"SR", 38}, {"Y", 39},
        {"ZR", 40}, {"NB", 41}, {"MO", 42}, {"TC", 43}, {"RU", 44},
        {"RH", 45}, {"PD", 46}, {"AG", 47}, {"CD", 48}, {"IN", 49},
        {"SN", 50}, {"SB", 51}, {"TE", 52}, {"I", 53},  {"XE", 54},
        {"CS", 55}, {"BA", 56}, {"LA", 57}, {"CE", 58}, {"PR", 59},
        {"ND", 60}, {"PM", 61}, {"SM", 62}, {"EU", 63}, {"GD", 64},
        {"TB", 65}, {"DY", 66}, {"HO", 67}, {"ER", 68}, {"TM", 69},
        {"YB", 70}, {"LU", 71}, {"HF", 72}, {"TA", 73}, {"W", 74},
        {"RE", 75}, {"OS", 76}, {"IR", 77}, {"PT", 78}, {"AU", 79},
        {"HG", 80}, {"TL", 81}, {"PB", 82}, {"BI", 83}, {"PO", 84},
        {"AT", 85}, {"RN", 86}, {"FR", 87}, {"RA", 88}, {"AC", 89},
        {"TH", 90}, {"PA", 91}, {"U", 92},  {"NP", 93}, {"PU", 94},
        {"AM", 95}, {"CM", 96}, {"BK", 97}, {"CF", 98}, {"ES", 99},
        {"FM", 100},{"MD", 101},{"NO", 102},{"LR", 103},{"RF", 104},
        {"DB", 105},{"SG", 106},{"BH", 107},{"HS", 108},{"MT", 109},
        {"DS", 110},{"RG", 111},{"CN", 112},{"NH", 113},{"FL", 114},
        {"MC", 115},{"LV", 116},{"TS", 117},{"OG", 118},
    };
    return table;
  }

  std::vector<Atom> atoms_;
  int multiplicity_{1};
  int charge_{0};
};

}  // namespace cuest
