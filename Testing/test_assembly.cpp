#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <filesystem>
#include "../Read_Mesh/mesh_reader_kokkos.hpp"
#include "../Element/element_stiffness.hpp"
#include "../Assembly/coo_to_csr.hpp"
#include "../Assembly/assemble_system.hpp"
#include "../Global/LoadVector.hpp"
#include "../Global/SparseMatrixCSR.hpp"

using namespace Catch::Matchers;

TEST_CASE("Assemble system from mesh", "[assembly]") {
    std::string mesh_file = std::filesystem::temp_directory_path() / "test_assembly_mesh.msh";
    std::ofstream out(mesh_file);
    out << R"($MeshFormat
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

    Mesh2D mesh = read_mesh(mesh_file);
    Kokkos::View<double*> f_elem("f_elem", mesh.num_elements);
    Kokkos::parallel_for("init_f", mesh.num_elements, KOKKOS_LAMBDA(int i) {
        f_elem(i) = 2.0;
    });

    SparseMatrixCSR K_global(mesh.num_nodes, mesh.num_nodes, 9);
    LoadVector F_global(mesh.num_nodes);
    F_global.zero();

    assemble_system(mesh, K_global, F_global, 1.0, f_elem);

    auto F_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), F_global.get_data());
    double total_force = F_host(0) + F_host(1) + F_host(2);
    REQUIRE_THAT(total_force, WithinAbs(1.0, 1e-12));

    REQUIRE(K_global.numRows == mesh.num_nodes);
    REQUIRE(K_global.numCols == mesh.num_nodes);

    std::filesystem::remove(mesh_file);
}
