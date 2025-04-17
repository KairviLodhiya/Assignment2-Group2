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
# Catch2 Tests
# --------------------------
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};$ENV{HOME}/catch2/install")
find_package(Catch2 3 REQUIRED)

add_executable(mesh_reader_test mesh_reader_test.cpp)
target_link_libraries(mesh_reader_test PRIVATE mesh_reader Catch2::Catch2WithMain)

add_executable(element_stiffness_test elementstiffness_main.cpp element_stiffness_test.cpp)
target_link_libraries(element_stiffness_test PRIVATE element_stiffness Catch2::Catch2)

add_executable(SparseMatrixTests
    test_sparse_matrix.cpp
    SparseMatrixCSR.cpp
)
add_executable(LoadVectorTests
    test_load_vector.cpp
    LoadVector.cpp
)

target_link_libraries(SparseMatrixTests
  PRIVATE
    Kokkos::kokkos
    Catch2::Catch2WithMain
)
target_link_libraries(LoadVectorTests
  PRIVATE
    Kokkos::kokkos
    Catch2::Catch2WithMain
)

target_include_directories(SparseMatrixTests PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/external/Catch2/src  # <- Needed for headers
)
target_include_directories(LoadVectorTests PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/external/Catch2/src
)

# --------------------------
# Read Mesh Integration Test
# --------------------------
add_executable(ReadMeshIntegrationTest
    test_read_mesh_integration.cpp
)

target_link_libraries(ReadMeshIntegrationTest
  PRIVATE
    mesh_reader
    element_stiffness
    Kokkos::kokkos
    Catch2::Catch2WithMain
)

target_include_directories(ReadMeshIntegrationTest PRIVATE
  ${CMAKE_SOURCE_DIR}
)
