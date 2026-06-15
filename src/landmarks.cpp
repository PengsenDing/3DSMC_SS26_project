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

bool is_correspondence_header_row(const std::vector<std::string>& fields) {
    return !fields.empty() && fields.front() == "bfm_landmark_name";
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

        if (fields.size() != 6 && fields.size() != 7) {
            throw std::runtime_error("Landmark CSV line " + std::to_string(line_number) +
                                     ": expected 6 or 7 columns");
        }

        const bool has_name_column = fields.size() == 7;
        const std::size_t coordinate_offset = has_name_column ? 1 : 0;
        Landmark2D landmark;
        landmark.index = parse_int_field(fields[0], line_number, "index");
        if (has_name_column) {
            landmark.name = fields[1];
        }
        landmark.u = parse_float_field(fields[1 + coordinate_offset], line_number, "u");
        landmark.v = parse_float_field(fields[2 + coordinate_offset], line_number, "v");
        landmark.z_norm = parse_float_field(fields[3 + coordinate_offset], line_number, "z_norm");
        landmark.x_norm = parse_float_field(fields[4 + coordinate_offset], line_number, "x_norm");
        landmark.y_norm = parse_float_field(fields[5 + coordinate_offset], line_number, "y_norm");
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
            return lhs.u < rhs.u;
        });
    auto y_range = std::minmax_element(
        landmarks.begin(), landmarks.end(), [](const Landmark2D& lhs, const Landmark2D& rhs) {
            return lhs.v < rhs.v;
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
    summary << "u range: [" << x_range.first->u << ", " << x_range.second->u << "]\n";
    summary << "v range: [" << y_range.first->v << ", " << y_range.second->v << "]\n";
    return summary.str();
}

std::vector<BfmMediaPipeCorrespondence> load_bfm_mediapipe_correspondences(
    const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open BFM/MediaPipe correspondence CSV: " +
                                 path.string());
    }

    std::vector<BfmMediaPipeCorrespondence> correspondences;
    std::string line;
    int line_number = 0;

    while (std::getline(input, line)) {
        ++line_number;
        if (trim(line).empty()) {
            continue;
        }

        const std::vector<std::string> fields = split_csv_row(line);
        if (line_number == 1 && is_correspondence_header_row(fields)) {
            continue;
        }

        if (fields.size() != 3) {
            throw std::runtime_error("BFM/MediaPipe correspondence CSV line " +
                                     std::to_string(line_number) + ": expected 3 columns");
        }

        BfmMediaPipeCorrespondence correspondence;
        correspondence.bfm_landmark_name = fields[0];
        correspondence.bfm_vertex_id = parse_int_field(fields[1], line_number, "bfm_vertex_id");
        correspondence.mediapipe_index =
            parse_int_field(fields[2], line_number, "mediapipe_index");
        correspondences.push_back(correspondence);
    }

    if (correspondences.empty()) {
        throw std::runtime_error("BFM/MediaPipe correspondence CSV has no rows: " +
                                 path.string());
    }

    return correspondences;
}

std::string bfm_mediapipe_correspondence_summary(
    const std::vector<BfmMediaPipeCorrespondence>& correspondences,
    const std::filesystem::path& source_path) {
    if (correspondences.empty()) {
        return "BFM/MediaPipe correspondences: <empty>\n";
    }

    auto bfm_vertex_range = std::minmax_element(
        correspondences.begin(), correspondences.end(),
        [](const BfmMediaPipeCorrespondence& lhs, const BfmMediaPipeCorrespondence& rhs) {
            return lhs.bfm_vertex_id < rhs.bfm_vertex_id;
        });
    auto mediapipe_index_range = std::minmax_element(
        correspondences.begin(), correspondences.end(),
        [](const BfmMediaPipeCorrespondence& lhs, const BfmMediaPipeCorrespondence& rhs) {
            return lhs.mediapipe_index < rhs.mediapipe_index;
        });

    std::ostringstream summary;
    summary << "BFM/MediaPipe correspondences: " << source_path.string() << '\n';
    summary << "Count: " << correspondences.size() << '\n';
    summary << "BFM vertex range: [" << bfm_vertex_range.first->bfm_vertex_id << ", "
            << bfm_vertex_range.second->bfm_vertex_id << "]\n";
    summary << "MediaPipe index range: [" << mediapipe_index_range.first->mediapipe_index << ", "
            << mediapipe_index_range.second->mediapipe_index << "]\n";
    return summary.str();
}

}  // namespace face_reconstruction
