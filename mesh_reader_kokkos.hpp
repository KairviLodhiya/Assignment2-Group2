#ifndef MESH_READER_KOKKOS_HPP
#define MESH_READER_KOKKOS_HPP

#include <Kokkos_Core.hpp>
#include <string>

struct Mesh2D {
    Kokkos::View<double**> node_coords;         // [num_nodes][2]
    Kokkos::View<int**> element_connectivity;   // [num_elements][nodes_per_elem]
    int num_nodes = 0;
    int num_elements = 0;
    int nodes_per_elem = 0;
};

Mesh2D read_mesh(const std::string& filename);

#endif // MESH_READER_KOKKOS_HPP
