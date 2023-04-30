cmake_minimum_required(VERSION 3.22)
project(cfdARCO LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

include(FetchContent)

find_package (OpenMP REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (indicators REQUIRED)
find_package(Matplot++ REQUIRED)
find_package(MPI REQUIRED)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)
FetchContent_Declare(argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
        GIT_TAG v2.9
        )
FetchContent_MakeAvailable(argparse)


set(CFDARCHO_BASE_SOURCES
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
        include/io_operators.hpp
        src/io_operators.cpp
        )

set(CFDARCHO_SOURCES ${CFDARCHO_BASE_SOURCES}
        src/main.cpp
        )

set(CFDARCHO_INFERENCE_SOURCES ${CFDARCHO_BASE_SOURCES}
        src/main_inference.cpp
        )

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    set(CFDARCHO_SOURCES ${CFDARCHO_SOURCES}
            src/cuda_operators.cu
            include/pool_allocator.hpp
            src/pool_allocator.cpp
            )
    set(CFDARCHO_INFERENCE_SOURCES ${CFDARCHO_INFERENCE_SOURCES}
            src/cuda_operators.cu
            include/pool_allocator.hpp
            src/pool_allocator.cpp
            )
    enable_language(CUDA)
    add_compile_definitions(CFDARCHO_CUDA_ENABLE)
endif()

add_executable(cfdARCO ${CFDARCHO_SOURCES})

target_include_directories(cfdARCO PRIVATE include)

target_link_libraries (cfdARCO Eigen3::Eigen)
target_link_libraries (cfdARCO OpenMP::OpenMP_CXX)
target_link_libraries (cfdARCO indicators::indicators)
target_link_libraries (cfdARCO Matplot++::matplot)
target_link_libraries(cfdARCO MPI::MPI_CXX)
target_link_libraries(cfdARCO nlohmann_json::nlohmann_json)
target_link_libraries(cfdARCO argparse)


add_executable(cfdARCO_inference ${CFDARCHO_INFERENCE_SOURCES})

target_include_directories(cfdARCO_inference PRIVATE include)

target_link_libraries (cfdARCO_inference Eigen3::Eigen)
target_link_libraries (cfdARCO_inference OpenMP::OpenMP_CXX)
target_link_libraries (cfdARCO_inference indicators::indicators)
target_link_libraries (cfdARCO_inference Matplot++::matplot)
target_link_libraries(cfdARCO_inference MPI::MPI_CXX)
target_link_libraries(cfdARCO_inference argparse)

if(CMAKE_CUDA_COMPILER)
    FetchContent_Declare(rmm
            GIT_REPOSITORY https://github.com/rapidsai/rmm.git
            GIT_TAG v23.04.00
            )
    FetchContent_MakeAvailable(rmm)
    target_link_libraries(cfdARCO rmm::rmm)
    set_target_properties( cfdARCO PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(cfdARCO_inference rmm::rmm)
    set_target_properties( cfdARCO_inference PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
