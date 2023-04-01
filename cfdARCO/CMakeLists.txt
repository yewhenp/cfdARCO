cmake_minimum_required(VERSION 3.25)
project(cfdARCO)

set(CMAKE_CXX_STANDARD 17)

find_package (OpenMP REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (indicators REQUIRED)
find_package(Matplot++ REQUIRED)
find_package(MPI REQUIRED)

add_executable(cfdARCO src/main.cpp include/abstract_mesh.hpp include/mesh2d.hpp src/mesh2d.cpp include/fvm.hpp src/fvm.cpp src/cfdarcho_main.cpp include/cfdarcho_main.hpp include/decls.hpp)

target_include_directories(cfdARCO PRIVATE include)

target_link_libraries (cfdARCO Eigen3::Eigen)
target_link_libraries (cfdARCO OpenMP::OpenMP_CXX)
target_link_libraries (cfdARCO indicators::indicators)
target_link_libraries (cfdARCO Matplot++::matplot)
target_link_libraries(cfdARCO MPI::MPI_CXX)