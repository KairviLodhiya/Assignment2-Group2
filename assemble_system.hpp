#ifndef ASSEMBLE_SYSTEM_HPP
#define ASSEMBLE_SYSTEM_HPP

inline void assemble_system(const Mesh2D& mesh, SparseMatrixCSR& K_global, LoadVector& F_global, double kappa, const Kokkos::View<double*>& f_elem) {
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

    auto Ke_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Ke);
    auto Fe_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Fe);
    auto conn_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh.element_connectivity);

    COOMatrix coo = create_coo_matrix(mesh.num_nodes, mesh.num_nodes);

    for (int e = 0; e < num_elems; ++e) {
        for (int i = 0; i < nodes_per_elem; ++i) {
            int row = conn_host(e, i);
            double Fi = Fe_host(e, i);
            F_global.add(row, Fi);

            for (int j = 0; j < nodes_per_elem; ++j) {
                int col = conn_host(e, j);
                double val = Ke_host(e, i, j);
                add_coo_entry(coo, row, col, val);
            }
        }
    }

    coo_to_csr(coo, K_global);
}

#endif // ASSEMBLE_SYSTEM_HPP