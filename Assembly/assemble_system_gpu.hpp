#ifndef ASSEMBLE_SYSTEM_GPU_HPP
#define ASSEMBLE_SYSTEM_GPU_HPP

#include "../Global/SparseMatrixCSR.hpp"
#include "../Global/LoadVector.hpp"
#include "../Element/create_element_coordinates.hpp"
#include "coo_to_csr_kokkos.hpp"
#include "../Element/element_stiffness.hpp"

inline void assemble_system_gpu(const Mesh2D& mesh, SparseMatrixCSR& K_global, LoadVector& F_global, double kappa, const Kokkos::View<double*>& f_elem) {
    const int num_elems = mesh.num_elements;
    const int nodes_per_elem = mesh.nodes_per_elem;

    ElementStiffness stiffness(kappa);
    auto element_coords = createElementCoordinates(mesh.node_coords, mesh.element_connectivity);

    Kokkos::View<double***> Ke("Ke", num_elems, nodes_per_elem, nodes_per_elem);
    Kokkos::View<double**> Fe("Fe", num_elems, nodes_per_elem);

    if (nodes_per_elem == 3) {
        stiffness.computeTriangleStiffnessKokkos(element_coords, Ke);
        stiffness.computeTriangleLoadVectorKokkos(element_coords, f_elem, Fe);
    } else if (nodes_per_elem == 4) {
        stiffness.computeQuadStiffnessKokkos(element_coords, Ke);
        stiffness.computeQuadLoadVectorKokkos(element_coords, f_elem, Fe);
    }

    // Prepare COO
    int max_nnz = num_elems * nodes_per_elem * nodes_per_elem;
    COOMatrixDevice coo = create_coo_matrix_device(mesh.num_nodes, mesh.num_nodes, max_nnz);
    Kokkos::View<int> coo_count("counter");

    Kokkos::parallel_for("AssembleCOO", num_elems, KOKKOS_LAMBDA(const int e) {
        for (int i = 0; i < nodes_per_elem; ++i) {
            int row = mesh.element_connectivity(e, i);
            double Fi = Fe(e, i);
            Kokkos::atomic_add(&F_global.get_data()(i), value);

            for (int j = 0; j < nodes_per_elem; ++j) {
                int col = mesh.element_connectivity(e, j);
                int idx = Kokkos::atomic_fetch_add(&coo_count(), 1);
                coo.row_idx(idx) = row;
                coo.col_idx(idx) = col;
                coo.values(idx) = Ke(e, i, j);
            }
        }
    });

    coo_to_csr_kokkos(coo, K_global);
}

#endif // ASSEMBLE_SYSTEM_GPU_HPP