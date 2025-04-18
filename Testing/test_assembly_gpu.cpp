#include <filesystem>
#include <fstream>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../ReadMesh/mesh_reader_kokkos.hpp"
#include "../Global/LoadVector.hpp"
#include "../Global/SparseMatrixCSR.hpp"
#include "../Element/element_stiffness.hpp"
#include "../Element/create_element_coordinates.hpp"
#include "../Assembly/assemble_system_gpu.hpp"

using namespace Catch::Matchers;

TEST_CASE("GPU assembly system from mesh", "[assembly][gpu]") {
    std::string mesh_file = std::filesystem::temp_directory_path() / "test_gpu_assembly.msh";
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

    SparseMatrixCSR K_global(mesh.num_nodes, mesh.num_nodes, mesh.num_nodes * mesh.nodes_per_elem);
    LoadVector F_global(mesh.num_nodes);
    F_global.zero();

    assemble_system_gpu(mesh, K_global, F_global, 1.0, f_elem);

    auto F_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), F_global.get_data());

    REQUIRE_THAT(F_host(0) + F_host(1) + F_host(2), WithinAbs(1.0, 1e-12));

    std::filesystem::remove(mesh_file);
}
