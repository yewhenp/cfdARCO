cmake_minimum_required(VERSION 3.25)
project(cfdARCO LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package (OpenMP REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (indicators REQUIRED)
find_package(Matplot++ REQUIRED)
find_package(MPI REQUIRED)

set(CFDARCHO_SOURCES
        src/main.cpp
        include/abstract_mesh.hpp
        include/mesh2d.hpp
        src/mesh2d.cpp
        include/fvm.hpp
        src/fvm.cpp
        src/cfdarcho_main.cpp
        include/cfdarcho_main.hpp
        include/decls.hpp
        include/distribution_algo.hpp
        src/distribution_algo.cpp
        include/cuda_operators.hpp
        src/cuda_operators.cu
        include/pool_allocator.hpp
        )

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    set(CFDARCHO_SOURCES ${CFDARCHO_SOURCES}
            src/cuda_operators.cu
            include/pool_allocator.hpp
            src/pool_allocator.cpp
            )
    enable_language(CUDA)
    dd_compile_definitions(CFDARCHO_CUDA_ENABLE)
endif()

add_executable(cfdARCO ${CFDARCHO_SOURCES})

target_include_directories(cfdARCO PRIVATE include)

target_link_libraries (cfdARCO Eigen3::Eigen)
target_link_libraries (cfdARCO OpenMP::OpenMP_CXX)
target_link_libraries (cfdARCO indicators::indicators)
target_link_libraries (cfdARCO Matplot++::matplot)
target_link_libraries(cfdARCO MPI::MPI_CXX)

if(CMAKE_CUDA_COMPILER)
    include(FetchContent)
    FetchContent_Declare(rmm
            GIT_REPOSITORY https://github.com/rapidsai/rmm.git
            GIT_TAG v23.04.00
            )
    FetchContent_MakeAvailable(rmm)
    target_link_libraries(cfdARCO rmm::rmm)
    set_target_properties( cfdARCO PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")