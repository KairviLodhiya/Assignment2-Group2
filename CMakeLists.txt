cmake_minimum_required(VERSION 3.28.3)
project(assignment2 CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------
# Kokkos Setup
# --------------------------
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};$ENV{HOME}/kokkos/install")
set(Kokkos_DIR "$ENV{HOME}/kokkos-ci-install/lib/cmake/Kokkos")
find_package(Kokkos REQUIRED)

# --------------------------
# Catch2 Setup
# --------------------------
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};$ENV{HOME}/catch2/install")
find_package(Catch2 3 REQUIRED)

# --------------------------
# Mesh Reader Library
# --------------------------
add_library(mesh_reader mesh_reader_kokkos.cpp)
target_include_directories(mesh_reader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mesh_reader PUBLIC Kokkos::kokkos)

# --------------------------
# Element Stiffness Library
# --------------------------
add_library(element_stiffness element_stiffness.cpp)
target_include_directories(element_stiffness PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(element_stiffness PUBLIC Kokkos::kokkos)

# --------------------------
# Sparse Matrix Library
# --------------------------
add_library(sparse_matrix_lib SparseMatrixCSR.cpp)
target_include_directories(sparse_matrix_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sparse_matrix_lib PUBLIC Kokkos::kokkos)

# --------------------------
# Load Vector Library
# --------------------------
add_library(load_vector_lib LoadVector.hpp)
target_include_directories(load_vector_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(load_vector_lib PUBLIC Kokkos::kokkos)

# --------------------------
# Combined Test Executable
# --------------------------
add_executable(all_tests
    test_main.cpp                 # Catch2 main() with Kokkos::initialize/finalize
    test_read_mesh_integration.cpp
    test_element_stiffness.cpp
    test_sparse_matrix.cpp
    test_load_vector.cpp
    test_mesh_reader.cpp
    test_assembly.cpp            
)

target_link_libraries(all_tests
    PRIVATE
    mesh_reader
    element_stiffness
    sparse_matrix_lib
    load_vector_lib
    Kokkos::kokkos
    Catch2::Catch2
)

target_include_directories(all_tests PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/external/Catch2/src
)

# --------------------------
# CTest Integration
# --------------------------
enable_testing()
add_test(NAME AllFEMTests COMMAND all_tests)
