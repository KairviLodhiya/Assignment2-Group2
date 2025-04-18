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
add_library(mesh_reader ReadMesh/mesh_reader_kokkos.cpp)
target_include_directories(mesh_reader PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(mesh_reader PUBLIC Kokkos::kokkos)

# --------------------------
# Element Stiffness Library
# --------------------------
add_library(element_stiffness Element/element_stiffness.cpp)
target_include_directories(element_stiffness PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(element_stiffness PUBLIC Kokkos::kokkos)

# --------------------------
# Sparse Matrix Library
# --------------------------
add_library(sparse_matrix_lib Global/SparseMatrixCSR.cpp)
target_include_directories(sparse_matrix_lib PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(sparse_matrix_lib PUBLIC Kokkos::kokkos)

# --------------------------
# Load Vector Header-Only Library
# --------------------------
add_library(load_vector_lib INTERFACE)
target_include_directories(load_vector_lib INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(load_vector_lib INTERFACE Kokkos::kokkos)

# --------------------------
# GPU COO to CSR (Device Assembly) Header-Only
# --------------------------
add_library(assembly_gpu_lib INTERFACE)
target_include_directories(assembly_gpu_lib INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(assembly_gpu_lib INTERFACE Kokkos::kokkos)

# --------------------------
# FEM Executable with Driver + Tests
# --------------------------
add_executable(fem_main
    Driver/main.cpp
    Testing/test_read_mesh_integration.cpp
    Testing/test_element_stiffness.cpp
    Testing/test_sparse_matrix.cpp
    Testing/test_load_vector.cpp
    Testing/test_mesh_reader.cpp
    Testing/test_assembly.cpp
)

target_link_libraries(fem_main
    PRIVATE
    mesh_reader
    element_stiffness
    sparse_matrix_lib
    load_vector_lib
    assembly_gpu_lib
    Kokkos::kokkos
    Catch2::Catch2
)

target_include_directories(fem_main PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/external/Catch2/src
)

# --------------------------
# CTest Integration (Optional)
# --------------------------
enable_testing()
add_test(NAME AllFEMTests COMMAND fem_main)
