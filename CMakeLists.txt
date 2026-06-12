cmake_minimum_required(VERSION 3.20)

project(
    face_reconstruction
    VERSION 0.1.0
    DESCRIPTION "Classical optimization based face reconstruction"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FACE_RECONSTRUCTION_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

find_package(PkgConfig REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Ceres CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

pkg_check_modules(OpenCV REQUIRED IMPORTED_TARGET opencv4)
pkg_check_modules(GLEW REQUIRED IMPORTED_TARGET glew)
pkg_check_modules(SFML_WINDOW REQUIRED IMPORTED_TARGET sfml-window)
pkg_check_modules(SFML_SYSTEM REQUIRED IMPORTED_TARGET sfml-system)

add_library(face_reconstruction_core
    src/app.cpp
    src/mesh.cpp
    src/obj_loader.cpp
    src/viewer.cpp
)

target_include_directories(face_reconstruction_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(face_reconstruction_core
    PUBLIC
        Eigen3::Eigen
        Ceres::ceres
        OpenGL::GL
        PkgConfig::OpenCV
        PkgConfig::GLEW
        PkgConfig::SFML_WINDOW
        PkgConfig::SFML_SYSTEM
)

target_compile_options(face_reconstruction_core
    PRIVATE
        $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall -Wextra -Wpedantic>
        $<$<AND:$<BOOL:${FACE_RECONSTRUCTION_WARNINGS_AS_ERRORS}>,$<CXX_COMPILER_ID:AppleClang,Clang,GNU>>:-Werror>
)

add_executable(face_reconstruction
    src/main.cpp
)

target_link_libraries(face_reconstruction
    PRIVATE
        face_reconstruction_core
)
