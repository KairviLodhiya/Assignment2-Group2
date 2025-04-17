#ifndef CREATE_ELEMENT_COORDINATES_HPP
#define CREATE_ELEMENT_COORDINATES_HPP

#include <Kokkos_Core.hpp>

// Converts global node coordinates and element connectivity into a 3D
// Kokkos View of per-element node coordinates for FEM computations.
inline Kokkos::View<double***> createElementCoordinates(
    const Kokkos::View<double**>& node_coords,
    const Kokkos::View<int**>& connectivity)
{
    const int num_elements = connectivity.extent(0);
    const int nodes_per_elem = connectivity.extent(1);

    Kokkos::View<double***> coords("element_coords", num_elements, nodes_per_elem, 2);

    Kokkos::parallel_for("AssembleElementCoords", num_elements, KOKKOS_LAMBDA(const int e) {
        for (int n = 0; n < nodes_per_elem; ++n) {
            int node = connectivity(e, n);
            coords(e, n, 0) = node_coords(node, 0); // x
            coords(e, n, 1) = node_coords(node, 1); // y
        }
    });

    return coords;
}

#endif // CREATE_ELEMENT_COORDINATES_HPP
