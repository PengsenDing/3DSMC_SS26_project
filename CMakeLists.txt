cmake_minimum_required(VERSION 3.20)

# HighFive 2.9.0 declares compatibility with an old CMake policy version.
# CMake 4 requires consumers to opt into a newer minimum for that dependency.
set(CMAKE_POLICY_VERSION_MINIMUM "3.5")

project(
    face_reconstruction
    VERSION 0.1.0
    DESCRIPTION "Classical optimization based 3D face reconstruction"
    LANGUAGES C CXX
)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FACE_RECONSTRUCTION_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# Allow the dependency bundle used by week_2 while still supporting system
# installations (for example Homebrew on macOS).
set(LIBRARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs" CACHE PATH "Optional local dependency directory")
if(EXISTS "${LIBRARY_DIR}/Eigen/share/eigen3/cmake")
    set(Eigen3_DIR "${LIBRARY_DIR}/Eigen/share/eigen3/cmake" CACHE PATH "Eigen CMake directory")
endif()
if(EXISTS "${LIBRARY_DIR}/Ceres/CMake")
    set(Ceres_DIR "${LIBRARY_DIR}/Ceres/CMake" CACHE PATH "Ceres CMake directory")
endif()
if(WIN32 AND EXISTS "${LIBRARY_DIR}/glog-lib/lib/cmake/glog")
    set(glog_DIR "${LIBRARY_DIR}/glog-lib/lib/cmake/glog" CACHE PATH "glog CMake directory")
endif()

find_package(PkgConfig REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Ceres CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(OpenGL REQUIRED)

pkg_check_modules(OpenCV REQUIRED IMPORTED_TARGET opencv4)
pkg_check_modules(GLEW REQUIRED IMPORTED_TARGET glew)
pkg_check_modules(SFML_WINDOW REQUIRED IMPORTED_TARGET sfml-window)
pkg_check_modules(SFML_SYSTEM REQUIRED IMPORTED_TARGET sfml-system)

include(FetchContent)

set(HIGHFIVE_USE_BOOST OFF CACHE BOOL "Disable Boost support in HighFive" FORCE)
set(HIGHFIVE_USE_EIGEN ON CACHE BOOL "Enable Eigen support in HighFive" FORCE)
set(HIGHFIVE_BUILD_DOCS OFF CACHE BOOL "Disable HighFive documentation" FORCE)
set(HIGHFIVE_EXAMPLES OFF CACHE BOOL "Disable HighFive examples" FORCE)
set(HIGHFIVE_UNIT_TESTS OFF CACHE BOOL "Disable HighFive tests" FORCE)

FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.4.2
)
FetchContent_Declare(
    highfive
    GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
    GIT_TAG v2.9.0
)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(cli11 highfive json)

function(face_reconstruction_enable_warnings target)
    target_compile_options(
        ${target}
        PRIVATE
            $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall -Wextra -Wpedantic>
            $<$<AND:$<BOOL:${FACE_RECONSTRUCTION_WARNINGS_AS_ERRORS}>,$<CXX_COMPILER_ID:AppleClang,Clang,GNU>>:-Werror>
    )
endfunction()

# Basel Face Model loader and inspection executable from week_2.
add_library(
    bfm_model
    face_recon/bfm_model.cc
    face_recon/bfm_model.h
)
target_include_directories(
    bfm_model
    PUBLIC
        "${PROJECT_SOURCE_DIR}"
        ${HDF5_INCLUDE_DIRS}
)
target_link_libraries(
    bfm_model
    PUBLIC
        Eigen3::Eigen
        glog::glog
        HighFive
        nlohmann_json::nlohmann_json
)
face_reconstruction_enable_warnings(bfm_model)

add_library(
    reconstruction_fitting
    face_recon/fitting.cc
    face_recon/fitting.h
    face_recon/image_fitting.cc
    face_recon/image_fitting.h
    face_recon/mesh_export.cc
    face_recon/mesh_export.h
    face_recon/photometric_fitting.cc
    face_recon/photometric_fitting.h
    face_recon/rasterizer.cc
    face_recon/rasterizer.h
    face_recon/texture_fitting.cc
    face_recon/texture_fitting.h
    src/landmarks.cpp
)
target_include_directories(
    reconstruction_fitting
    PUBLIC
        "${PROJECT_SOURCE_DIR}"
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(
    reconstruction_fitting
    PUBLIC
        bfm_model
        Ceres::ceres
        Eigen3::Eigen
        PkgConfig::OpenCV
)
face_reconstruction_enable_warnings(reconstruction_fitting)

add_executable(face_reconstruction face_recon/main.cc)
target_link_libraries(
    face_reconstruction
    PRIVATE
        reconstruction_fitting
        CLI11::CLI11
)
face_reconstruction_enable_warnings(face_reconstruction)

if(BUILD_TESTING)
    add_executable(fitting_test tests/fitting_test.cpp)
    target_link_libraries(fitting_test PRIVATE reconstruction_fitting)
    face_reconstruction_enable_warnings(fitting_test)
    add_test(NAME fitting_test COMMAND fitting_test)

    add_executable(rasterizer_test tests/rasterizer_test.cpp)
    target_link_libraries(rasterizer_test PRIVATE reconstruction_fitting)
    face_reconstruction_enable_warnings(rasterizer_test)
    add_test(NAME rasterizer_test COMMAND rasterizer_test)

    add_executable(photometric_fitting_test tests/photometric_fitting_test.cpp)
    target_link_libraries(photometric_fitting_test PRIVATE reconstruction_fitting)
    face_reconstruction_enable_warnings(photometric_fitting_test)
    add_test(NAME photometric_fitting_test COMMAND photometric_fitting_test)

    add_executable(texture_fitting_test tests/texture_fitting_test.cpp)
    target_link_libraries(texture_fitting_test PRIVATE reconstruction_fitting)
    face_reconstruction_enable_warnings(texture_fitting_test)
    add_test(NAME texture_fitting_test COMMAND texture_fitting_test)
endif()

# OBJ viewer and MediaPipe landmark CSV tooling from landmark_detection.
add_library(
    landmark_viewer_core
    src/app.cpp
    src/landmarks.cpp
    src/mesh.cpp
    src/obj_loader.cpp
    src/ply_loader.cpp
    src/viewer.cpp
)
target_include_directories(
    landmark_viewer_core
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(
    landmark_viewer_core
    PUBLIC
        Eigen3::Eigen
        Ceres::ceres
        OpenGL::GL
        PkgConfig::OpenCV
        PkgConfig::GLEW
        PkgConfig::SFML_WINDOW
        PkgConfig::SFML_SYSTEM
)
face_reconstruction_enable_warnings(landmark_viewer_core)

add_executable(landmark_viewer src/main.cpp)
target_link_libraries(landmark_viewer PRIVATE landmark_viewer_core)
face_reconstruction_enable_warnings(landmark_viewer)

if(WIN32)
    set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY VS_STARTUP_PROJECT face_reconstruction)
endif()
