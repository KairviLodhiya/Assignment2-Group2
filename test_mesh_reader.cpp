#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <filesystem>
#include "mesh_reader_kokkos.hpp" 

using namespace Catch::Matchers;

TEST_CASE("read_mesh loads small triangle mesh", "[mesh]") {
    std::string mesh_filename = std::filesystem::temp_directory_path() / "test_mesh.msh";

    std::ofstream out(mesh_filename);
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
    {
        Mesh2D mesh = read_mesh(mesh_filename);

        REQUIRE(mesh.num_nodes == 3);
        REQUIRE(mesh.num_elements == 1);
        REQUIRE(mesh.nodes_per_elem == 3);

        auto coords_host = Kokkos::create_mirror_view(mesh.node_coords);
        Kokkos::deep_copy(coords_host, mesh.node_coords);

        CHECK_THAT(coords_host(0, 0), WithinAbs(0.0, 1e-12));
        CHECK_THAT(coords_host(1, 0), WithinAbs(1.0, 1e-12));
        CHECK_THAT(coords_host(2, 1), WithinAbs(1.0, 1e-12));
    }

    std::filesystem::remove(mesh_filename);
}
