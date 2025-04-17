#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <string>
#include <filesystem>

#include "mesh_reader_kokkos.hpp"
#include "create_element_coordinates.hpp"
#include "element_stiffness.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("read_mesh + stiffness/load vector integration", "[fem][mesh]") {
    Kokkos::initialize();
    {
        // Write minimal triangle mesh to a temporary file
        std::string mesh_file = std::filesystem::temp_directory_path() / "test_mesh_triangle.msh";
        std::ofstream out(mesh_file);
        out << R"(
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
3
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.0 1.0 0.0
$EndNodes
$Elements
1
1 2 1 0 1 2 3
$EndElements
)";
        out.close();

        // Read mesh
        Mesh2D mesh = read_mesh(mesh_file);
        REQUIRE(mesh.num_nodes == 3);
        REQUIRE(mesh.num_elements == 1);
        REQUIRE(mesh.nodes_per_elem == 3);

        // Generate element-local coordinates
        auto element_coords = createElementCoordinates(mesh.node_coords, mesh.element_connectivity);

        // Allocate output views
        Kokkos::View<double***> K("K", mesh.num_elements, 3, 3);
        Kokkos::View<double**> Fe("Fe", mesh.num_elements, 3);
        Kokkos::View<double*> f("f", mesh.num_elements);

        Kokkos::parallel_for("init_f", mesh.num_elements, KOKKOS_LAMBDA(const int i) {
            f(i) = 2.0; // constant force
        });

        // Compute stiffness and load
        ElementStiffness stiffness(1.0);
        stiffness.computeTriangleStiffnessKokkos(element_coords, K);
        stiffness.computeTriangleLoadVectorKokkos(element_coords, f, Fe);

        // Mirror to host
        auto K_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), K);
        auto Fe_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Fe);

        SECTION("Stiffness matrix is symmetric and nonzero") {
            double sum = 0.0;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    sum += std::abs(K_host(0, i, j));
                    // Symmetry check using matcher
                    REQUIRE_THAT(K_host(0, i, j), WithinAbs(K_host(0, j, i), 1e-12));
                }
            }
            REQUIRE(sum > 0.0);
        }

        SECTION("Load vector sums to expected total") {
            double total = 0.0;
            for (int i = 0; i < 3; ++i) {
                total += Fe_host(0, i);
            }
            REQUIRE_THAT(total, WithinAbs(1.0, 1e-12)); // area = 0.5, f = 2 → total = 1
        }
    }
    Kokkos::finalize();
}
