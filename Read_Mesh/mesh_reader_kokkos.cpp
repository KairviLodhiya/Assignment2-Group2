#include "../Read Mesh/mesh_reader_kokkos.hpp"
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

Mesh2D read_mesh(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Cannot open mesh file.");

    Mesh2D mesh;
    std::vector<double> host_coords; // flat [x0, y0, x1, y1, ...]
    std::vector<int> host_connectivity;

    std::string line;
    while (std::getline(file, line)) {
        if (line == "$Nodes") {
            int n;
            file >> n;
            mesh.num_nodes = n;
            host_coords.resize(2 * n);
            for (int i = 0; i < n; ++i) {
                int id;
                double x, y, z;
                file >> id >> x >> y >> z;
                host_coords[2 * (id - 1) + 0] = x;
                host_coords[2 * (id - 1) + 1] = y;
            }
            std::getline(file, line); // flush
            std::getline(file, line); // $EndNodes
        }
        if (line == "$Elements") {
            int total;
            file >> total;
            std::vector<std::vector<int>> elements;
            for (int i = 0; i < total; ++i) {
                int id, type, num_tags;
                file >> id >> type >> num_tags;
                for (int j = 0; j < num_tags; ++j) {
                    int skip; file >> skip;
                }

                int nn = (type == 2) ? 3 : (type == 3) ? 4 : 0;
                if (nn == 0) continue;

                if (mesh.nodes_per_elem == 0) mesh.nodes_per_elem = nn;
                else if (mesh.nodes_per_elem != nn) throw std::runtime_error("Mixed element types not supported.");

                std::vector<int> conn(nn);
                for (int j = 0; j < nn; ++j) {
                    file >> conn[j];
                    conn[j] -= 1; // zero-based indexing
                }
                elements.push_back(conn);
            }

            mesh.num_elements = static_cast<int>(elements.size());
            host_connectivity.resize(mesh.num_elements * mesh.nodes_per_elem);
            for (int e = 0; e < mesh.num_elements; ++e) {
                for (int j = 0; j < mesh.nodes_per_elem; ++j) {
                    host_connectivity[e * mesh.nodes_per_elem + j] = elements[e][j];
                }
            }

            std::getline(file, line); // flush
            std::getline(file, line); // $EndElements
        }
    }

    // Allocate and populate views
    mesh.node_coords = Kokkos::View<double**>("node_coords", mesh.num_nodes, 2);
    mesh.element_connectivity = Kokkos::View<int**>("connectivity", mesh.num_elements, mesh.nodes_per_elem);

    // Parallel copy node coordinates
    Kokkos::parallel_for("CopyNodes", mesh.num_nodes, KOKKOS_LAMBDA(const int i) {
        mesh.node_coords(i, 0) = host_coords[2 * i];
        mesh.node_coords(i, 1) = host_coords[2 * i + 1];
    });

    // Parallel copy connectivity
    Kokkos::parallel_for("CopyConnectivity", mesh.num_elements, KOKKOS_LAMBDA(const int e) {
        for (int j = 0; j < mesh.nodes_per_elem; ++j) {
            mesh.element_connectivity(e, j) = host_connectivity[e * mesh.nodes_per_elem + j];
        }
    });

    return mesh;
}
