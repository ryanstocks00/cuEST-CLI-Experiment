#pragma once
/**
 * @file json_parser.hpp
 * @brief Minimal recursive-descent JSON parser for BSE basis set data.
 *
 * Supports: objects, arrays, strings, numbers, booleans, null.
 * Numbers are always stored as strings to preserve exact precision,
 * with helper functions to convert to double/int64_t as needed.
 */

#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuest {
namespace json {

// ---------------------------------------------------------------------------
// JSON value type
// ---------------------------------------------------------------------------
struct Value;

using Object = std::map<std::string, Value>;
using Array  = std::vector<Value>;

struct Value {
    enum Kind { Null, Bool, Number, String, Object, Array };
    Kind kind = Null;

    bool   bool_val = false;
    std::string number_val;   // stored as string for precision
    std::string string_val;
    ::cuest::json::Object  object_val;
    ::cuest::json::Array   array_val;

    Value() = default;
    Value(Kind k) : kind(k) {}
    explicit Value(bool b) : kind(Bool), bool_val(b) {}
    explicit Value(const std::string& s) : kind(String), string_val(s) {}
    explicit Value(std::string&& s) : kind(String), string_val(std::move(s)) {}
    explicit Value(const char* s) : kind(String), string_val(s) {}
    explicit Value(double d) : kind(Number), number_val(std::to_string(d)) {}

    static Value number(const std::string& s) { Value v; v.kind=Number; v.number_val=s; return v; }
    static Value object(::cuest::json::Object&& o) { Value v; v.kind=Value::Object; v.object_val=std::move(o); return v; }
    static Value array(::cuest::json::Array&& a)  { Value v; v.kind=Value::Array;  v.array_val=std::move(a);  return v; }

    // Access helpers
    const Value& operator[](const std::string& key) const;
    const Value& operator[](size_t idx) const;
    size_t size() const;

    // Numeric conversions (handles both Number and String kinds, since BSE
    // stores numeric values as JSON strings for precision)
    double as_double() const {
        const std::string& s = (kind == Number) ? number_val : string_val;
        std::string cleaned = s;
        for (auto& c : cleaned)
            if (c == 'D' || c == 'd') c = 'E';  // fortran double
        return std::stod(cleaned);
    }
    int64_t as_int() const {
        return static_cast<int64_t>(as_double());
    }
    uint64_t as_uint() const {
        return static_cast<uint64_t>(as_double());
    }
    bool     as_bool()     const { return bool_val; }
    const std::string& as_string() const { return string_val; }

    bool is_object() const { return kind == Object; }
    bool is_array()  const { return kind == Array; }
    bool is_number() const { return kind == Number; }
    bool is_string() const { return kind == String; }
    bool is_bool()   const { return kind == Bool; }

    bool has(const std::string& key) const {
        return kind == Object && object_val.find(key) != object_val.end();
    }
};

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------
class Parser {
public:
    explicit Parser(const std::string& src) : src_(src), pos_(0) {}

    Value parse() {
        skip_ws();
        Value v = parse_value();
        skip_ws();
        if (pos_ < src_.size())
            throw std::runtime_error("JSON: trailing content at position " + std::to_string(pos_));
        return v;
    }

private:
    const std::string& src_;
    size_t pos_;

    char peek() const { return pos_ < src_.size() ? src_[pos_] : '\0'; }
    char next() { return pos_ < src_.size() ? src_[pos_++] : '\0'; }
    void expect(char c) {
        if (next() != c)
            throw std::runtime_error(std::string("JSON: expected '") + c + "' at " + std::to_string(pos_));
    }

    void skip_ws() {
        while (pos_ < src_.size() && (src_[pos_] == ' ' || src_[pos_] == '\t' ||
               src_[pos_] == '\n' || src_[pos_] == '\r'))
            pos_++;
    }

    Value parse_value() {
        skip_ws();
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return Value(parse_string());
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        if (c == '-' || (c >= '0' && c <= '9')) return Value::number(parse_number());
        throw std::runtime_error(std::string("JSON: unexpected char '") + c + "' at " + std::to_string(pos_));
    }

    Value parse_object() {
        expect('{');
        skip_ws();
        Object obj;
        if (peek() == '}') { next(); return Value::object(std::move(obj)); }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            expect(':');
            obj[key] = parse_value();
            skip_ws();
            char c = next();
            if (c == '}') break;
            if (c != ',') throw std::runtime_error(std::string("JSON: expected ',' or '}' in object at ") + std::to_string(pos_));
        }
        return Value::object(std::move(obj));
    }

    Value parse_array() {
        expect('[');
        skip_ws();
        Array arr;
        if (peek() == ']') { next(); return Value::array(std::move(arr)); }
        while (true) {
            arr.push_back(parse_value());
            skip_ws();
            char c = next();
            if (c == ']') break;
            if (c != ',') throw std::runtime_error(std::string("JSON: expected ',' or ']' in array at ") + std::to_string(pos_));
        }
        return Value::array(std::move(arr));
    }

    std::string parse_string() {
        expect('"');
        std::string s;
        while (true) {
            char c = next();
            if (c == '"') break;
            if (c == '\\') {
                c = next();
                switch (c) {
                    case '"':  s += '"';  break;
                    case '\\': s += '\\'; break;
                    case '/':  s += '/';  break;
                    case 'b':  s += '\b'; break;
                    case 'f':  s += '\f'; break;
                    case 'n':  s += '\n'; break;
                    case 'r':  s += '\r'; break;
                    case 't':  s += '\t'; break;
                    case 'u': {
                        // Read 4 hex digits
                        uint16_t cp = 0;
                        for (int i = 0; i < 4; i++) {
                            char h = next();
                            cp <<= 4;
                            if (h >= '0' && h <= '9') cp |= (h - '0');
                            else if (h >= 'a' && h <= 'f') cp |= (h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') cp |= (h - 'A' + 10);
                            else throw std::runtime_error("JSON: invalid unicode escape");
                        }
                        // Simple UTF-8 encode (BMP only)
                        if (cp < 0x80) { s += static_cast<char>(cp); }
                        else if (cp < 0x800) {
                            s += static_cast<char>(0xC0 | (cp >> 6));
                            s += static_cast<char>(0x80 | (cp & 0x3F));
                        } else {
                            s += static_cast<char>(0xE0 | (cp >> 12));
                            s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            s += static_cast<char>(0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default:
                        throw std::runtime_error(std::string("JSON: invalid escape \\") + c);
                }
            } else {
                s += c;
            }
        }
        return s;
    }

    std::string parse_number() {
        size_t start = pos_;
        if (peek() == '-') next();
        // Integer part
        if (peek() == '0') { next(); }
        else if (peek() >= '1' && peek() <= '9') {
            while (peek() >= '0' && peek() <= '9') next();
        } else {
            throw std::runtime_error("JSON: invalid number at " + std::to_string(pos_));
        }
        // Fraction
        if (peek() == '.') {
            next();
            if (peek() < '0' || peek() > '9')
                throw std::runtime_error("JSON: expected digit after '.' at " + std::to_string(pos_));
            while (peek() >= '0' && peek() <= '9') next();
        }
        // Exponent
        if (peek() == 'e' || peek() == 'E') {
            next();
            if (peek() == '+' || peek() == '-') next();
            if (peek() < '0' || peek() > '9')
                throw std::runtime_error("JSON: expected digit in exponent at " + std::to_string(pos_));
            while (peek() >= '0' && peek() <= '9') next();
        }
        return src_.substr(start, pos_ - start);
    }

    Value parse_bool() {
        if (peek() == 't') {
            for (auto c : {'t','r','u','e'}) expect(c);
            return Value(true);
        }
        for (auto c : {'f','a','l','s','e'}) expect(c);
        return Value(false);
    }

    Value parse_null() {
        for (auto c : {'n','u','l','l'}) expect(c);
        return Value();
    }
};

// ---------------------------------------------------------------------------
// Value member implementations
// ---------------------------------------------------------------------------
inline const Value& Value::operator[](const std::string& key) const {
    auto it = object_val.find(key);
    if (it == object_val.end())
        throw std::runtime_error("JSON: key '" + key + "' not found");
    return it->second;
}

inline const Value& Value::operator[](size_t idx) const {
    if (idx >= array_val.size())
        throw std::runtime_error("JSON: index " + std::to_string(idx) + " out of range");
    return array_val[idx];
}

inline size_t Value::size() const {
    if (kind == Object) return object_val.size();
    if (kind == Array)  return array_val.size();
    return 0;
}

// ---------------------------------------------------------------------------
// Convenience: parse a JSON file
// ---------------------------------------------------------------------------
inline Value parse_file(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("Cannot open JSON file: " + path);
    std::string content((std::istreambuf_iterator<char>(fin)),
                         std::istreambuf_iterator<char>());
    return Parser(content).parse();
}

}  // namespace json
}  // namespace cuest
