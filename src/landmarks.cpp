#include "face_reconstruction/landmarks.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace face_reconstruction {
namespace {

std::string trim(std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
        text.remove_prefix(1);
    }

    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
        text.remove_suffix(1);
    }

    return std::string(text);
}

std::vector<std::string> split_csv_row(const std::string& line) {
    std::vector<std::string> fields;
    std::istringstream stream(line);
    std::string field;

    while (std::getline(stream, field, ',')) {
        fields.push_back(trim(field));
    }

    return fields;
}

int parse_int_field(const std::string& value, int line_number, std::string_view field_name) {
    try {
        std::size_t parsed_chars = 0;
        const int result = std::stoi(value, &parsed_chars);
        if (parsed_chars != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return result;
    } catch (const std::exception&) {
        throw std::runtime_error("Landmark CSV line " + std::to_string(line_number) +
                                 ": invalid integer field " + std::string(field_name));
    }
}

float parse_float_field(const std::string& value, int line_number, std::string_view field_name) {
    try {
        std::size_t parsed_chars = 0;
        const float result = std::stof(value, &parsed_chars);
        if (parsed_chars != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return result;
    } catch (const std::exception&) {
        throw std::runtime_error("Landmark CSV line " + std::to_string(line_number) +
                                 ": invalid float field " + std::string(field_name));
    }
}

bool is_header_row(const std::vector<std::string>& fields) {
    return !fields.empty() && fields.front() == "index";
}

}  // namespace

std::vector<Landmark2D> load_landmarks_csv(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open landmark CSV: " + path.string());
    }

    std::vector<Landmark2D> landmarks;
    std::string line;
    int line_number = 0;

    while (std::getline(input, line)) {
        ++line_number;
        if (trim(line).empty()) {
            continue;
        }

        const std::vector<std::string> fields = split_csv_row(line);
        if (line_number == 1 && is_header_row(fields)) {
            continue;
        }

        if (fields.size() != 6) {
            throw std::runtime_error("Landmark CSV line " + std::to_string(line_number) +
                                     ": expected 6 columns");
        }

        Landmark2D landmark;
        landmark.index = parse_int_field(fields[0], line_number, "index");
        landmark.x_px = parse_float_field(fields[1], line_number, "x_px");
        landmark.y_px = parse_float_field(fields[2], line_number, "y_px");
        landmark.z_norm = parse_float_field(fields[3], line_number, "z_norm");
        landmark.x_norm = parse_float_field(fields[4], line_number, "x_norm");
        landmark.y_norm = parse_float_field(fields[5], line_number, "y_norm");
        landmarks.push_back(landmark);
    }

    if (landmarks.empty()) {
        throw std::runtime_error("Landmark CSV has no landmarks: " + path.string());
    }

    return landmarks;
}

std::string landmark_summary(const std::vector<Landmark2D>& landmarks,
                             const std::filesystem::path& source_path) {
    if (landmarks.empty()) {
        return "Landmarks: <empty>\n";
    }

    auto x_range = std::minmax_element(
        landmarks.begin(), landmarks.end(), [](const Landmark2D& lhs, const Landmark2D& rhs) {
            return lhs.x_px < rhs.x_px;
        });
    auto y_range = std::minmax_element(
        landmarks.begin(), landmarks.end(), [](const Landmark2D& lhs, const Landmark2D& rhs) {
            return lhs.y_px < rhs.y_px;
        });
    auto index_range = std::minmax_element(
        landmarks.begin(), landmarks.end(), [](const Landmark2D& lhs, const Landmark2D& rhs) {
            return lhs.index < rhs.index;
        });

    std::ostringstream summary;
    summary << "Landmarks: " << source_path.string() << '\n';
    summary << "Count: " << landmarks.size() << '\n';
    summary << "Index range: [" << index_range.first->index << ", " << index_range.second->index
            << "]\n";
    summary << std::fixed << std::setprecision(2);
    summary << "X pixel range: [" << x_range.first->x_px << ", " << x_range.second->x_px
            << "]\n";
    summary << "Y pixel range: [" << y_range.first->y_px << ", " << y_range.second->y_px
            << "]\n";
    return summary.str();
}

}  // namespace face_reconstruction
