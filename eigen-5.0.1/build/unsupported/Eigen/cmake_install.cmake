# Install script for directory: /Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/Libs/Eigen")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/unsupported/Eigen/AdolcForward"
    "/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/unsupported/Eigen/AlignedVector3"
    "/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/unsupported/Eigen/ArpackSupport"
    "/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/unsupported/Eigen/AutoDiff"
    "/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/unsupported/Eigen/BVH"
    "/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/unsupported/Eigen/EulerAngles"
    "/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/unsupported/Eigen/FFT"
    "/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/unsupported/Eigen/IterativeSolvers"
    "/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/unsupported/Eigen/KroneckerProduct"
    "/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/unsupported/Eigen/LevenbergMarquardt"
    "/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/unsupported/Eigen/MatrixFunctions"
    "/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/unsupported/Eigen/MPRealSupport"
    "/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/unsupported/Eigen/NNLS"
    "/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/unsupported/Eigen/NonLinearOptimization"
    "/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/unsupported/Eigen/NumericalDiff"
    "/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/unsupported/Eigen/OpenGLSupport"
    "/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/unsupported/Eigen/Polynomials"
    "/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/unsupported/Eigen/SparseExtra"
    "/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/unsupported/Eigen/SpecialFunctions"
    "/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/unsupported/Eigen/Splines"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/build/unsupported/Eigen/CXX11/cmake_install.cmake")

endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/dingpengsen/Desktop/IN2354 3DSMC/Exercises/eigen-5.0.1/build/unsupported/Eigen/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
