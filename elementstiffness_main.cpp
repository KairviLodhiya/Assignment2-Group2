#include <catch2/catch_session.hpp>
#include <Kokkos_Core.hpp>

int main (int argc, char* argv[]) {
    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    
    // Run Catch2 tests
    int result = Catch::Session().run(argc, argv);
    
    // Finalize Kokkos
    Kokkos::finalize();
    
    return result;
}