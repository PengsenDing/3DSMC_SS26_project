cmake_minimum_required(VERSION 3.14)

# Silence legacy policy errors from external dependencies (HighFive/CLI11).
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()
# Set a global minimum policy version to bypass the < 3.5 fatal error.
set(CMAKE_POLICY_VERSION_MINIMUM "3.14")

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Allowed build types" FORCE)
project(face_reconstruction)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options.
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
set(Ceres_DIR ${LIBRARY_DIR}/Ceres/CMake CACHE PATH "Path to installed Ceres")
set(Flann_INCLUDE_DIR ${LIBRARY_DIR}/Flann-1.8.4/ CACHE PATH "Path to Flann source folder")
# Dual Compatibility Switch for Glog.
if(WIN32)
    set(glog_DIR ${LIBRARY_DIR}/glog-lib/lib/cmake/glog CACHE PATH "Path to installed glog")
else()
    # On Linux, CMake will use the users's global glog.
    set(glog_DIR "/usr/lib/cmake/glog" CACHE PATH "Path to installed glog")
endif(WIN32)

# Use pre-compiled libraries on Windows and on Linux.
if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x64/ CACHE PATH "Path to FreeImage header file")
        set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x64/ CACHE PATH "Path to FreeImage .lib/.dll folder")
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x32/ CACHE PATH "Path to FreeImage header file")
        set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x32/ CACHE PATH "Path to FreeImage .lib/.dll folder")
    endif()
else()
    # Linux Setup.
    set(FreeImage_INCLUDE_DIR "/usr/include")
    set(FreeImage_LIBRARIES "freeimage") # Linker automatically finds /usr/lib/libfreeimage.so.
endif(WIN32)

# Set C++ flags - C++17 (HighFive requires modern C++ traits).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Ceres REQUIRED)
find_package(glog REQUIRED)
get_target_property(GLOG_DLL_PATH_DEBUG glog::glog IMPORTED_LOCATION_DEBUG)
get_target_property(GLOG_DLL_PATH_RELEASE glog::glog IMPORTED_LOCATION_RELEASE)

# Automatic External Library Management (No manual setup needed).
include(FetchContent)

# Fetch CLI11 for command line option handling.
FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.4.2
)
FetchContent_MakeAvailable(cli11)

# Find or build the core HDF5 system dependency.
# (On Linux/Mac it uses system HDF5; on Windows it falls back gracefully or uses vcpkg).
find_package(HDF5 REQUIRED COMPONENTS C CXX)
# Tell HighFive we are using Eigen, not Boost.
# Define the exact path to local Eigen header files.
set(LOCAL_EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/Eigen/include/eigen3")

# Hardcode variables using $<BUILD_INTERFACE:...> to bypass the generation error.
set(Eigen3_INCLUDE_DIRS "$<BUILD_INTERFACE:${LOCAL_EIGEN_INCLUDE_DIR}>" CACHE STRING "TUM Eigen Headers" FORCE)
set(EIGEN3_INCLUDE_DIR  "$<BUILD_INTERFACE:${LOCAL_EIGEN_INCLUDE_DIR}>" CACHE STRING "TUM Eigen Headers" FORCE)
set(EIGEN3_INCLUDE_DIRS "$<BUILD_INTERFACE:${LOCAL_EIGEN_INCLUDE_DIR}>" CACHE STRING "TUM Eigen Headers" FORCE)
set(Eigen3_FOUND TRUE CACHE BOOL "Force Eigen Found" FORCE)
set(EIGEN3_FOUND TRUE CACHE BOOL "Force Eigen Found" FORCE)

# Explicitly create the required modern CMake target HighFive links against.
if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
    target_include_directories(Eigen3::Eigen INTERFACE "${LOCAL_EIGEN_INCLUDE_DIR}")
endif()

# Standard HighFive configuration toggles.
set(HIGHFIVE_USE_BOOST OFF CACHE INTERNAL "Disable Boost backend for HighFive")
set(HIGHFIVE_USE_EIGEN ON CACHE INTERNAL "Enable Eigen backend for HighFive")
set(HIGHFIVE_BUILD_DOCS OFF CACHE INTERNAL "Disable documentation building")
set(HIGHFIVE_EXAMPLES OFF CACHE INTERNAL "Disable building HighFive examples")
set(HIGHFIVE_UNIT_TESTS OFF CACHE INTERNAL "Disable building HighFive tests")

# Fetch HighFive (Native Eigen to HDF5 binder).
FetchContent_Declare(
    highfive
    GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
    GIT_TAG        v2.9.0
)
FetchContent_MakeAvailable(highfive)

# Fetch nlohmann/json for metadata string decoding.
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(json)

add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

set(HEADER_FILES
    face_recon/bfm_model.h
)

set(SOURCE_FILES 
    face_recon/bfm_model.cc
    face_recon/main.cc
)

link_directories(${FreeImage_LIBRARY_DIR})
add_executable(face_reconstruction ${SOURCE_FILES} ${HEADER_FILES})
target_link_libraries(face_reconstruction ceres ${FreeImage_LIBRARIES} Eigen3::Eigen glog::glog HighFive CLI11::CLI11 nlohmann_json::nlohmann_json) 
target_include_directories(face_reconstruction PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR} ${Flann_INCLUDE_DIR} ${HDF5_INCLUDE_DIRS})

# On Windows copy dll to executable folder.
if(WIN32)
    # Visual Studio properties.
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT face_reconstruction)
    set_property(TARGET face_reconstruction PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

    # On Windows copy dll to executable folder.
    add_custom_command(TARGET face_reconstruction POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FreeImage_LIBRARY_DIR}/FreeImage.dll"
            ${CMAKE_BINARY_DIR})
    
    add_custom_command(TARGET face_reconstruction POST_BUILD        
        COMMAND ${CMAKE_COMMAND} -E copy_if_different          
            "${GLOG_DLL_PATH_DEBUG}"                           
            ${CMAKE_BINARY_DIR})                     
    
    add_custom_command(TARGET face_reconstruction POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GLOG_DLL_PATH_RELEASE}"
            ${CMAKE_BINARY_DIR})
endif(WIN32)
