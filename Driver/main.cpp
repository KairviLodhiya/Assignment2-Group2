#include <Kokkos_Core.hpp>
#include <iostream>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <catch2/catch_session.hpp>

#include "../ReadMesh/mesh_reader_kokkos.hpp"
#include "../Element/element_stiffness.hpp"
#include "../Assembly/coo_to_csr.hpp"
#include "../Assembly/assemble_system.hpp"
#include "../Global/LoadVector.hpp"
#include "../Global/SparseMatrixCSR.hpp"
#include "../Element/create_element_coordinates.hpp"

int run_driver(const std::string& mesh_file, const std::string& force_expr) {
    std::unordered_map<std::string, std::function<double(double, double)>> force_dispatch = {
        {"1", [](double, double) { return 1.0; }},
        {"x", [](double x, double) { return x; }},
        {"y", [](double, double y) { return y; }},
        {"xy", [](double x, double y) { return x * y; }},
        {"x2y", [](double x, double y) { return x * x * y; }},
        {"x+y", [](double x, double y) { return x + y; }},
        {"sinxy", [](double x, double y) { return std::sin(x * y); }}
    };

    if (!std::filesystem::exists(mesh_file)) {
        std::cerr << "Error: Mesh file '" << mesh_file << "' not found.\n";
        return 1;
    }

    if (!force_dispatch.contains(force_expr)) {
        std::cerr << "Error: Unknown force expression '" << force_expr << "'\nAvailable options:";
        for (const auto& [key, _] : force_dispatch) {
            std::cerr << " " << key;
        }
        std::cerr << "\n";
        return 1;
    }

    auto force_func = force_dispatch[force_expr];
    Mesh2D mesh = read_mesh(mesh_file);

    Kokkos::View<double*> f_elem("f_elem", mesh.num_elements);
    auto coords_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh.node_coords);
    auto conn_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mesh.element_connectivity);

    for (int e = 0; e < mesh.num_elements; ++e) {
        double xc = 0.0, yc = 0.0;
        for (int j = 0; j < mesh.nodes_per_elem; ++j) {
            int nid = conn_host(e, j);
            xc += coords_host(nid, 0);
            yc += coords_host(nid, 1);
        }
        xc /= mesh.nodes_per_elem;
        yc /= mesh.nodes_per_elem;
        f_elem(e) = force_func(xc, yc);
    }

    SparseMatrixCSR K_global(mesh.num_nodes, mesh.num_nodes, mesh.num_nodes * mesh.nodes_per_elem);
    LoadVector F_global(mesh.num_nodes);
    F_global.zero();

    assemble_system(mesh, K_global, F_global, 1.0, f_elem);

    auto F_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), F_global.get_data());

    std::cout << "\nLoad vector (assembled):\n";
    for (int i = 0; i < mesh.num_nodes; ++i) {
        std::cout << "F[" << i << "] = " << F_host(i) << "\n";
    }

    return 0;
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    int result = 0;

    // Determine if we should run Catch2 tests
    bool run_tests = (argc == 1);
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.starts_with("--") || arg.starts_with("[")) {
            run_tests = true;
            break;
        }
    }

    if (run_tests) {
        result = Catch::Session().run(argc, argv);
    } else if (argc >= 3) {
        result = run_driver(argv[1], argv[2]);
    } else {
        std::cerr << "Usage:\n"
                  << "  To run driver: " << argv[0] << " <mesh_file.msh> <force_expr>\n"
                  << "  To run tests : " << argv[0] << " --reporter console --success\n";
        result = 1;
    }

    Kokkos::finalize();
    return result;
}
