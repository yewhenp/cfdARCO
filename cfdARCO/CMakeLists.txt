cmake_minimum_required(VERSION 3.25)
project(cfdARCO LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package (OpenMP REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (indicators REQUIRED)
find_package(Matplot++ REQUIRED)
find_package(MPI REQUIRED)

add_executable(cfdARCO src/main.cpp include/abstract_mesh.hpp include/mesh2d.hpp src/mesh2d.cpp include/fvm.hpp
        src/fvm.cpp src/cfdarcho_main.cpp include/cfdarcho_main.hpp include/decls.hpp include/distribution_algo.hpp
        src/distribution_algo.cpp include/cuda_operators.hpp src/cuda_operators.cu include/pool_allocator.hpp)

target_include_directories(cfdARCO PRIVATE include)

include(FetchContent)
FetchContent_Declare(rmm
        GIT_REPOSITORY https://github.com/rapidsai/rmm.git
        GIT_TAG v23.04.00
        )
FetchContent_MakeAvailable(rmm)

target_link_libraries (cfdARCO Eigen3::Eigen)
target_link_libraries (cfdARCO OpenMP::OpenMP_CXX)
target_link_libraries (cfdARCO indicators::indicators)
target_link_libraries (cfdARCO Matplot++::matplot)
target_link_libraries(cfdARCO MPI::MPI_CXX)
target_link_libraries(cfdARCO rmm::rmm)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

set_target_properties( cfdARCO PROPERTIES CUDA_SEPARABLE_COMPILATION ON)