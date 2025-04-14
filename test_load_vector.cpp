#include <catch2/catch_test_macros.hpp>
#include "LoadVector.cpp"
#include <catch2/catch_approx.hpp>
#include <Kokkos_Core.hpp>

using Catch::Approx;

TEST_CASE("LoadVector basic operations", "[LoadVector]") {
    Kokkos::initialize();
    {
        const int N = 5;
        LoadVector lv(N);

        // Step 1: Zero the vector
        lv.zero();

        // Copy to host to check values
        auto host_view = Kokkos::create_mirror_view(lv.get_data());
        Kokkos::deep_copy(host_view, lv.get_data());

        for (int i = 0; i < N; ++i) {
            REQUIRE(host_view(i) == 0.0);
        }

        // Step 2: Add 3.5 to index 2
        lv.add(2, 3.5);
        Kokkos::deep_copy(host_view, lv.get_data());

        for (int i = 0; i < N; ++i) {
            if (i == 2)
                REQUIRE(host_view(i) == Approx(3.5));
            else
                REQUIRE(host_view(i) == 0.0);
        }
    }
    Kokkos::finalize();
}
