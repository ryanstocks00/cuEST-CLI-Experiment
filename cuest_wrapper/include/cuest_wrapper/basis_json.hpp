#pragma once
/**
 * @file basis_json.hpp
 * @brief BSE (Basis Set Exchange) JSON format reader (nlohmann/json).
 *
 * Parses https://www.basissetexchange.org/api/basis/<name>/format/json/
 * JSON format which contains electron_shells and optional ecp_potentials
 * for each element, keyed by atomic number.
 *
 * BSE often stores numeric fields as JSON strings (for precision / Fortran
 * D-exponents); helpers below accept either string or number.
 */

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "molecule.hpp"

namespace cuest {

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Numeric helpers (BSE: number-or-string)
// ---------------------------------------------------------------------------
inline double json_as_double(const json& v) {
    if (v.is_number())
        return v.get<double>();
    if (v.is_string()) {
        std::string s = v.get<std::string>();
        for (auto& c : s) {
            if (c == 'D' || c == 'd')
                c = 'E';  // Fortran double
        }
        return std::stod(s);
    }
    throw std::runtime_error("JSON value is neither number nor string");
}

inline uint64_t json_as_uint(const json& v) {
    return static_cast<uint64_t>(json_as_double(v));
}

inline json load_json_file(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot open JSON file: " + path);
    json j;
    in >> j;
    return j;
}

// ---------------------------------------------------------------------------
// Parsed shell from JSON
// ---------------------------------------------------------------------------
struct JsonShell {
    uint64_t L;                     // angular momentum
    uint64_t nprim;                 // number of primitives
    std::vector<double> exponents;
    std::vector<std::vector<double>> all_coefficients;  // one per contraction
};

// ---------------------------------------------------------------------------
// Parsed ECP data from JSON
// ---------------------------------------------------------------------------
struct JsonECP {
    uint64_t n_elec;                           // number of core electrons
    std::vector<uint64_t> shell_types;         // angular momentum per shell
    std::vector<uint64_t> num_primitives;
    std::vector<uint64_t> Ns;                  // r-exponents per primitive
    std::vector<double> exponents;
    std::vector<double> coefficients;
};

// ---------------------------------------------------------------------------
// BSE JSON basis set reader
// ---------------------------------------------------------------------------
class BSEJsonReader {
public:
    explicit BSEJsonReader(const std::string& json_path)
        : data_(load_json_file(json_path)) {
        if (!data_.contains("elements") || !data_["elements"].is_object())
            throw std::runtime_error("BSE JSON missing 'elements' object");
    }

    std::vector<JsonShell> get_shells(int atomic_number) const;
    std::vector<JsonShell> get_shells(const std::string& symbol) const {
        return get_shells(Molecule::symbol_to_z(symbol));
    }

    bool has_ecp(int atomic_number) const;
    bool has_ecp(const std::string& symbol) const {
        return has_ecp(Molecule::symbol_to_z(symbol));
    }

    JsonECP get_ecp(int atomic_number) const;
    JsonECP get_ecp(const std::string& symbol) const {
        return get_ecp(Molecule::symbol_to_z(symbol));
    }

private:
    json data_;
};

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------
inline std::vector<JsonShell> BSEJsonReader::get_shells(int atomic_number) const {
    const std::string z_str = std::to_string(atomic_number);
    const auto& elements = data_["elements"];

    if (!elements.contains(z_str))
        throw std::runtime_error("Element Z=" + z_str + " not found in basis set JSON");

    const auto& elem = elements[z_str];
    if (!elem.contains("electron_shells") || !elem["electron_shells"].is_array())
        throw std::runtime_error("Element Z=" + z_str + " has no electron_shells in JSON");

    std::vector<JsonShell> result;

    for (const auto& sh : elem["electron_shells"]) {
        if (!sh.contains("angular_momentum") || !sh.contains("exponents") ||
            !sh.contains("coefficients"))
            throw std::runtime_error("BSE shell missing required fields for Z=" + z_str);

        // angular_momentum: [L] for a single-L shell (possibly generally
        // contracted), or [0,1] / [0,1,2] for SP / SPD shells where
        // coefficients[i] belongs to angular_momentum[i].
        const auto& am = sh["angular_momentum"];
        if (!am.is_array() || am.empty())
            throw std::runtime_error("BSE shell has empty angular_momentum for Z=" + z_str);

        const auto& exps = sh["exponents"];
        if (!exps.is_array())
            throw std::runtime_error("BSE shell exponents must be an array for Z=" + z_str);
        const uint64_t nprim = exps.size();

        const auto& coeff_arrays = sh["coefficients"];
        if (!coeff_arrays.is_array() || coeff_arrays.empty())
            throw std::runtime_error("BSE shell has empty coefficients for Z=" + z_str);

        std::vector<double> shared_exps;
        shared_exps.reserve(nprim);
        for (const auto& e : exps)
            shared_exps.push_back(json_as_double(e));

        auto push_shell = [&](uint64_t L, const json& coeff_arr) {
            if (!coeff_arr.is_array() || coeff_arr.size() != nprim)
                throw std::runtime_error(
                    "BSE shell coefficient count mismatch for Z=" + z_str);

            JsonShell js;
            js.L = L;
            js.nprim = nprim;
            js.exponents = shared_exps;

            std::vector<double> this_coeffs;
            this_coeffs.reserve(nprim);
            for (const auto& c : coeff_arr)
                this_coeffs.push_back(json_as_double(c));
            js.all_coefficients.push_back(std::move(this_coeffs));
            result.push_back(std::move(js));
        };

        if (am.size() == 1) {
            const uint64_t L = json_as_uint(am[0]);
            for (const auto& coeff_arr : coeff_arrays)
                push_shell(L, coeff_arr);
        } else {
            if (am.size() != coeff_arrays.size())
                throw std::runtime_error(
                    "BSE multi-L shell: angular_momentum/coefficients size "
                    "mismatch for Z=" + z_str);
            for (size_t i = 0; i < am.size(); i++)
                push_shell(json_as_uint(am[i]), coeff_arrays[i]);
        }
    }

    return result;
}

inline bool BSEJsonReader::has_ecp(int atomic_number) const {
    const std::string z_str = std::to_string(atomic_number);
    const auto& elements = data_["elements"];
    if (!elements.contains(z_str))
        return false;
    return elements[z_str].contains("ecp_potentials");
}

inline JsonECP BSEJsonReader::get_ecp(int atomic_number) const {
    const std::string z_str = std::to_string(atomic_number);
    const auto& elements = data_["elements"];
    if (!elements.contains(z_str))
        throw std::runtime_error("Element Z=" + z_str + " not found in basis set JSON");

    const auto& elem = elements[z_str];
    if (!elem.contains("ecp_potentials"))
        throw std::runtime_error("Element Z=" + z_str + " has no ECP data in JSON");

    JsonECP ecp;
    ecp.n_elec = elem.contains("ecp_electrons") ? json_as_uint(elem["ecp_electrons"]) : 0;

    for (const auto& pot : elem["ecp_potentials"]) {
        if (!pot.contains("angular_momentum") || !pot.contains("r_exponents") ||
            !pot.contains("gaussian_exponents") || !pot.contains("coefficients"))
            throw std::runtime_error("BSE ECP potential missing required fields");

        const uint64_t L = json_as_uint(pot["angular_momentum"].at(0));

        const auto& r_exps = pot["r_exponents"];
        const auto& g_exps = pot["gaussian_exponents"];
        const auto& coeffs = pot["coefficients"].at(0);

        const uint64_t nprim = r_exps.size();
        ecp.shell_types.push_back(L);
        ecp.num_primitives.push_back(nprim);

        for (size_t i = 0; i < nprim; i++) {
            ecp.Ns.push_back(json_as_uint(r_exps.at(i)));
            ecp.exponents.push_back(json_as_double(g_exps.at(i)));
            ecp.coefficients.push_back(json_as_double(coeffs.at(i)));
        }
    }

    return ecp;
}

}  // namespace cuest
