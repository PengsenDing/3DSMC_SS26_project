#include "face_reconstruction/app.hpp"

#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        return face_reconstruction::run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
