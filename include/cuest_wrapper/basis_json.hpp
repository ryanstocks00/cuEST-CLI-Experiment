#pragma once
/**
 * @file basis_json.hpp
 * @brief BSE (Basis Set Exchange) JSON format reader.
 *
 * Parses https://www.basissetexchange.org/api/basis/<name>/format/json/
 * JSON format which contains electron_shells and optional ecp_potentials
 * for each element, keyed by atomic number.
 */

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "json_parser.hpp"
#include "molecule.hpp"

namespace cuest {

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
        : data_(json::parse_file(json_path)) {
        if (!data_.has("elements"))
            throw std::runtime_error("BSE JSON missing 'elements' key");
    }

    // Get electron shells for an element (by atomic number or symbol)
    std::vector<JsonShell> get_shells(int atomic_number) const;
    std::vector<JsonShell> get_shells(const std::string& symbol) const {
        return get_shells(Molecule::symbol_to_z(symbol));
    }

    // Check if ECP data exists for this element
    bool has_ecp(int atomic_number) const;
    bool has_ecp(const std::string& symbol) const {
        return has_ecp(Molecule::symbol_to_z(symbol));
    }

    // Get ECP data for an element
    JsonECP get_ecp(int atomic_number) const;
    JsonECP get_ecp(const std::string& symbol) const {
        return get_ecp(Molecule::symbol_to_z(symbol));
    }

private:
    json::Value data_;
};

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------
inline std::vector<JsonShell> BSEJsonReader::get_shells(int atomic_number) const {
    std::string z_str = std::to_string(atomic_number);
    const auto& elements = data_["elements"];

    std::vector<JsonShell> result;

    // Handle "element references" in BSE: element may redirect to another entry
    const json::Value* elem = nullptr;
    if (elements.has(z_str)) {
        elem = &elements[z_str];
    } else {
        // Some BSE entries use element references by symbol
        for (const auto& [key, val] : elements.object_val) {
            if (val.has("element_references")) {
                // Not implemented: would need to follow references
            }
        }
        throw std::runtime_error("Element Z=" + z_str + " not found in basis set JSON");
    }

    if (!elem->has("electron_shells"))
        throw std::runtime_error("Element Z=" + z_str + " has no electron_shells in JSON");

    for (const auto& shell_val : (*elem)["electron_shells"].array_val) {
        const auto& sh = shell_val;
        if (!sh.has("angular_momentum") || !sh.has("exponents") || !sh.has("coefficients"))
            throw std::runtime_error("BSE shell missing required fields for Z=" + z_str);

        // angular_momentum is an array of integers
        const auto& am = sh["angular_momentum"].array_val;
        uint64_t L = am[0].as_uint();

        // exponents are stored as string numbers
        const auto& exps = sh["exponents"].array_val;
        uint64_t nprim = exps.size();

        // coefficients is array of arrays: one per contraction
        const auto& coeff_arrays = sh["coefficients"].array_val;
        if (coeff_arrays.empty())
            throw std::runtime_error("BSE shell has empty coefficients for Z=" + z_str);

        // Parse exponents once (shared across contractions)
        std::vector<double> shared_exps;
        shared_exps.reserve(nprim);
        for (size_t i = 0; i < nprim; i++)
            shared_exps.push_back(exps[i].as_double());

        // Create one JsonShell per coefficient array (general contraction)
        for (const auto& coeff_arr : coeff_arrays) {
            const auto& coeffs = coeff_arr.array_val;

            JsonShell js;
            js.L = L;
            js.nprim = nprim;
            js.exponents = shared_exps;  // shared exponents

            if (coeffs.size() != nprim)
                throw std::runtime_error(
                    "BSE shell coefficient count mismatch for Z=" + z_str);
            std::vector<double> this_coeffs;
            this_coeffs.reserve(nprim);
            for (size_t i = 0; i < nprim; i++)
                this_coeffs.push_back(coeffs[i].as_double());
            js.all_coefficients.push_back(std::move(this_coeffs));

            result.push_back(std::move(js));
        }
    }

    return result;
}

inline bool BSEJsonReader::has_ecp(int atomic_number) const {
    std::string z_str = std::to_string(atomic_number);
    const auto& elements = data_["elements"];
    if (!elements.has(z_str)) return false;
    return elements[z_str].has("ecp_potentials");
}

inline JsonECP BSEJsonReader::get_ecp(int atomic_number) const {
    std::string z_str = std::to_string(atomic_number);
    const auto& elements = data_["elements"];
    if (!elements.has(z_str))
        throw std::runtime_error("Element Z=" + z_str + " not found in basis set JSON");

    const auto& elem = elements[z_str];
    if (!elem.has("ecp_potentials"))
        throw std::runtime_error("Element Z=" + z_str + " has no ECP data in JSON");

    uint64_t n_elec = elem.has("ecp_electrons") ? elem["ecp_electrons"].as_uint() : 0;

    JsonECP ecp;
    ecp.n_elec = n_elec;

    for (const auto& pot_val : elem["ecp_potentials"].array_val) {
        const auto& pot = pot_val;
        if (!pot.has("angular_momentum") || !pot.has("r_exponents") ||
            !pot.has("gaussian_exponents") || !pot.has("coefficients"))
            throw std::runtime_error("BSE ECP potential missing required fields");

        uint64_t L = pot["angular_momentum"].array_val[0].as_uint();

        const auto& r_exps = pot["r_exponents"].array_val;
        const auto& g_exps = pot["gaussian_exponents"].array_val;
        const auto& coeffs = pot["coefficients"].array_val[0].array_val;

        uint64_t nprim = r_exps.size();
        ecp.shell_types.push_back(L);
        ecp.num_primitives.push_back(nprim);

        for (size_t i = 0; i < nprim; i++) {
            ecp.Ns.push_back(r_exps[i].as_uint());
            ecp.exponents.push_back(g_exps[i].as_double());
            ecp.coefficients.push_back(coeffs[i].as_double());
        }
    }

    return ecp;
}

}  // namespace cuest
