cmake_minimum_required(VERSION 3.25)
project(cfdARCO)

set(CMAKE_CXX_STANDARD 17)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(cfdARCO main.cpp include/abstract_mesh.hpp include/mesh2d.hpp src/mesh2d.cpp include/fvm.hpp src/fvm.cpp)

target_include_directories(cfdARCO PRIVATE include)

target_link_libraries (cfdARCO Eigen3::Eigen)
