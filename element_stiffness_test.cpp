#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <Kokkos_Core.hpp>
#include "element_stiffness.hpp"

TEST_CASE("Triangle stiffness matrix", "[ElementStiffness]") {
    
    // Create a sample triangle with vertices at (0,0), (1,0), and (0,1)
    const int numElements = 1;
    Kokkos::View<double***> coords("coords", numElements, 3, 2);
    Kokkos::View<double***> K("K", numElements, 3, 3);
    
    // Initialize host view for coordinates
    auto h_coords = Kokkos::create_mirror_view(coords);
    
    // Triangle with vertices at (0,0), (1,0), (0,1)
    h_coords(0, 0, 0) = 0.0; h_coords(0, 0, 1) = 0.0; // Node 1 (x1, y1)
    h_coords(0, 1, 0) = 1.0; h_coords(0, 1, 1) = 0.0; // Node 2 (x2, y2)
    h_coords(0, 2, 0) = 0.0; h_coords(0, 2, 1) = 1.0; // Node 3 (x3, y3)
    
    // Copy host data to device
    Kokkos::deep_copy(coords, h_coords);
    
    // Define material property and create ElementStiffness object
    double kappa = 1.0;
    ElementStiffness stiffness(kappa);
    
    // Compute stiffness matrix
    stiffness.computeTriangleStiffnessKokkos(coords, K);
    
    // Copy results back to host for verification
    auto h_K = Kokkos::create_mirror_view(K);
    Kokkos::deep_copy(h_K, K);
    
    // Expected stiffness matrix
    double expected[3][3] = {
        { 2.0, -1.0, -1.0},
        {-1.0,  1.0,  0.0},
        {-1.0,  0.0,  1.0}
    };
    
    // Check results
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            REQUIRE(h_K(0, i, j) == Catch::Approx(expected[i][j]).margin(1e-8));
        }
    }

}

TEST_CASE("Quad stiffness matrix", "[ElementStiffness]") {
     
    // Create a sample quad with vertices at (0,0), (1,0), (1,1), and (0,1)
    const int numElements = 1;
    Kokkos::View<double***> coords("coords", numElements, 4, 2);
    Kokkos::View<double***> K("K", numElements, 4, 4);
    
    // Initialize host view for coordinates
    auto h_coords = Kokkos::create_mirror_view(coords);
    
    // Square with vertices at (0,0), (1,0), (1,1), and (0,1)
    h_coords(0, 0, 0) = 0.0; h_coords(0, 0, 1) = 0.0; // Node 1 (x1, y1)
    h_coords(0, 1, 0) = 1.0; h_coords(0, 1, 1) = 0.0; // Node 2 (x2, y2)
    h_coords(0, 2, 0) = 1.0; h_coords(0, 2, 1) = 1.0; // Node 3 (x3, y3)
    h_coords(0, 3, 0) = 0.0; h_coords(0, 3, 1) = 1.0; // Node 4 (x4, y4)
    
    // Copy host data to device
    Kokkos::deep_copy(coords, h_coords);
    
    // Define material property and create ElementStiffness object
    double kappa = 1.0;
    ElementStiffness stiffness(kappa);
    
    // Compute stiffness matrix
    stiffness.computeQuadStiffnessKokkos(coords, K);
    
    // Copy results back to host for verification
    auto h_K = Kokkos::create_mirror_view(K);
    Kokkos::deep_copy(h_K, K);
    
    // Expected stiffness matrix
    double expected[4][4] = {
        { 2.0/3.0, -1.0/6.0, -1.0/3.0, -1.0/6.0},
        {-1.0/6.0,  2.0/3.0, -1.0/6.0, -1.0/3.0},
        {-1.0/3.0, -1.0/6.0,  2.0/3.0, -1.0/6.0},
        {-1.0/6.0, -1.0/3.0, -1.0/6.0,  2.0/3.0}
    };
    
    // Check results
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            REQUIRE(h_K(0, i, j) == Catch::Approx(expected[i][j]).margin(1e-8));
        }
    }

}

TEST_CASE("Triangle load vector", "[ElementStiffness]") {
    
    // Create a sample triangle with vertices at (0,0), (1,0), and (0,1)
    const int numElements = 1;
    Kokkos::View<double***> coords("coords", numElements, 3, 2);
    Kokkos::View<double*> f("f", numElements);
    Kokkos::View<double**> Fe("Fe", numElements, 3);
    
    // Initialize host views
    auto h_coords = Kokkos::create_mirror_view(coords);
    auto h_f = Kokkos::create_mirror_view(f);
    
    // Triangle with vertices at (0,0), (1,0), (0,1)
    h_coords(0, 0, 0) = 0.0; h_coords(0, 0, 1) = 0.0; // Node 1 (x1, y1)
    h_coords(0, 1, 0) = 1.0; h_coords(0, 1, 1) = 0.0; // Node 2 (x2, y2)
    h_coords(0, 2, 0) = 0.0; h_coords(0, 2, 1) = 1.0; // Node 3 (x3, y3)
    h_f(0) = 1.0; // Force value
    
    // Copy host data to device
    Kokkos::deep_copy(coords, h_coords);
    Kokkos::deep_copy(f, h_f);
    
    // Define material property and create ElementStiffness object
    double kappa = 1.0;
    ElementStiffness stiffness(kappa);
    
    // Compute load vector
    stiffness.computeTriangleLoadVectorKokkos(coords, f, Fe);
    
    // Copy results back to host for verification
    auto h_Fe = Kokkos::create_mirror_view(Fe);
    Kokkos::deep_copy(h_Fe, Fe);
    
    // Expected load vector (area = 0.5, f = 1.0)
    double expected[3] = {1.0/6.0, 1.0/6.0, 1.0/6.0};
    
    // Check results
    for (int i = 0; i < 3; i++) {
        REQUIRE(h_Fe(0, i) == Catch::Approx(expected[i]).margin(1e-10));
    }
    
}

TEST_CASE("Quad load vector", "[ElementStiffness]") {
    
    // Create a sample quad with vertices at (0,0), (1,0), (1,1), and (0,1)
    const int numElements = 1;
    Kokkos::View<double***> coords("coords", numElements, 4, 2);
    Kokkos::View<double*> f("f", numElements);
    Kokkos::View<double**> Fe("Fe", numElements, 4);
    
    // Initialize host views
    auto h_coords = Kokkos::create_mirror_view(coords);
    auto h_f = Kokkos::create_mirror_view(f);
    
    // Square with vertices at (0,0), (1,0), (1,1), and (0,1)
    h_coords(0, 0, 0) = 0.0; h_coords(0, 0, 1) = 0.0; // Node 1 (x1, y1)
    h_coords(0, 1, 0) = 1.0; h_coords(0, 1, 1) = 0.0; // Node 2 (x2, y2)
    h_coords(0, 2, 0) = 1.0; h_coords(0, 2, 1) = 1.0; // Node 3 (x3, y3)
    h_coords(0, 3, 0) = 0.0; h_coords(0, 3, 1) = 1.0; // Node 4 (x4, y4)
    h_f(0) = 1.0; // Force value
    
    // Copy host data to device
    Kokkos::deep_copy(coords, h_coords);
    Kokkos::deep_copy(f, h_f);
    
    // Define material property and create ElementStiffness object
    double kappa = 1.0;
    ElementStiffness stiffness(kappa);
    
    // Compute load vector
    stiffness.computeQuadLoadVectorKokkos(coords, f, Fe);
    
    // Copy results back to host for verification
    auto h_Fe = Kokkos::create_mirror_view(Fe);
    Kokkos::deep_copy(h_Fe, Fe);
    
    // Expected load vector (area = 1.0, f = 1.0)
    // For a unit square with 2x2 Gauss quadrature, each node gets 1/4 of the load
    double expected[4] = {0.25, 0.25, 0.25, 0.25};
    
    // Check results
    for (int i = 0; i < 4; i++) {
        REQUIRE(h_Fe(0, i) == Catch::Approx(expected[i]).margin(1e-10));
    }
    
}

TEST_CASE("Triangle stiffness matrix for multiple elements", "[ElementStiffness]") {
    
    // Create two triangles
    const int numElements = 2;
    Kokkos::View<double***> coords("coords", numElements, 3, 2);
    Kokkos::View<double***> K("K", numElements, 3, 3);
    
    // Initialize host view for coordinates
    auto h_coords = Kokkos::create_mirror_view(coords);
    
    // First triangle with vertices at (0,0), (1,0), (0,1)
    h_coords(0, 0, 0) = 0.0; h_coords(0, 0, 1) = 0.0;
    h_coords(0, 1, 0) = 1.0; h_coords(0, 1, 1) = 0.0;
    h_coords(0, 2, 0) = 0.0; h_coords(0, 2, 1) = 1.0;
    
    // Second triangle with vertices at (1,0), (1,1), (0,1)
    h_coords(1, 0, 0) = 1.0; h_coords(1, 0, 1) = 0.0;
    h_coords(1, 1, 0) = 1.0; h_coords(1, 1, 1) = 1.0;
    h_coords(1, 2, 0) = 0.0; h_coords(1, 2, 1) = 1.0;
    
    // Copy host data to device
    Kokkos::deep_copy(coords, h_coords);
    
    // Define material property and create ElementStiffness object
    double kappa = 1.0;
    ElementStiffness stiffness(kappa);
    
    // Compute stiffness matrix
    stiffness.computeTriangleStiffnessKokkos(coords, K);
    
    // Copy results back to host for verification
    auto h_K = Kokkos::create_mirror_view(K);
    Kokkos::deep_copy(h_K, K);
    
    // Expected stiffness matrix for first triangle
    double expected1[3][3] = {
        { 2.0, -1.0, -1.0},
        {-1.0,  1.0,  0.0},
        {-1.0,  0.0,  1.0}
    };
    
    // Expected stiffness matrix for second triangle
    double expected2[3][3] = {
        { 1.0,  0.0, -1.0},
        { 0.0,  1.0, -1.0},
        {-1.0, -1.0,  2.0}
    };
    
    // Check results for first triangle
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            REQUIRE(h_K(0, i, j) == Catch::Approx(expected1[i][j]).margin(1e-8));
        }
    }
    
    // Check results for second triangle
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            REQUIRE(h_K(1, i, j) == Catch::Approx(expected2[i][j]).margin(1e-8));
        }
    }
    
}

TEST_CASE("Quad stiffness matrix for irregular quadrilateral", "[ElementStiffness]") {
    
    // Create an irregular quadrilateral
    const int numElements = 1;
    Kokkos::View<double***> coords("coords", numElements, 4, 2);
    Kokkos::View<double***> K("K", numElements, 4, 4);
    
    // Initialize host view for coordinates
    auto h_coords = Kokkos::create_mirror_view(coords);
    
    // Irregular quadrilateral with vertices
    h_coords(0, 0, 0) = 0.0; h_coords(0, 0, 1) = 0.0;  // Node 1 (x1, y1)
    h_coords(0, 1, 0) = 2.0; h_coords(0, 1, 1) = 0.0;  // Node 2 (x2, y2)
    h_coords(0, 2, 0) = 2.0; h_coords(0, 2, 1) = 1.0;  // Node 3 (x3, y3)
    h_coords(0, 3, 0) = 0.0; h_coords(0, 3, 1) = 2.0;  // Node 4 (x4, y4)
    
    // Copy host data to device
    Kokkos::deep_copy(coords, h_coords);
    
    // Define material property and create ElementStiffness object
    double kappa = 1.0;
    ElementStiffness stiffness(kappa);
    
    // Compute stiffness matrix
    stiffness.computeQuadStiffnessKokkos(coords, K);
    
    // Copy results back to host for verification
    auto h_K = Kokkos::create_mirror_view(K);
    Kokkos::deep_copy(h_K, K);
    
    // For irregular quadrilaterals, we just check some basic properties
    // 1. The matrix should be symmetric
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            REQUIRE(h_K(0, i, j) == Catch::Approx(h_K(0, j, i)).margin(1e-8));
        }
    }
    
    // 2. The sum of each row should be approximately zero
    for (int i = 0; i < 4; i++) {
        double rowSum = 0.0;
        for (int j = 0; j < 4; j++) {
            rowSum += h_K(0, i, j);
        }
        REQUIRE(rowSum == Catch::Approx(0.0).margin(1e-8));
    }
    
    // 3. All diagonal elements should be positive
    for (int i = 0; i < 4; i++) {
        REQUIRE(h_K(0, i, i) > 0.0);
    }
    
    // 4. All off-diagonal elements should be negative or zero
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i != j) {
                REQUIRE(h_K(0, i, j) <= Catch::Approx(0.0).margin(1e-8));
            }
        }
    }
    
}

TEST_CASE("Element load vectors with varying force values", "[ElementStiffness]") {
    
    // Create multiple elements
    const int numElements = 2;
    Kokkos::View<double***> coords("coords", numElements, 3, 2);
    Kokkos::View<double*> f("f", numElements);
    Kokkos::View<double**> Fe("Fe", numElements, 3);
    
    // Initialize host views
    auto h_coords = Kokkos::create_mirror_view(coords);
    auto h_f = Kokkos::create_mirror_view(f);
    
    // Two triangles with different force values
    // First triangle: vertices at (0,0), (1,0), (0,1) with f = 2.0
    h_coords(0, 0, 0) = 0.0; h_coords(0, 0, 1) = 0.0;
    h_coords(0, 1, 0) = 1.0; h_coords(0, 1, 1) = 0.0;
    h_coords(0, 2, 0) = 0.0; h_coords(0, 2, 1) = 1.0;
    h_f(0) = 2.0;
    
    // Second triangle: vertices at (1,0), (1,1), (0,1) with f = 3.0
    h_coords(1, 0, 0) = 1.0; h_coords(1, 0, 1) = 0.0;
    h_coords(1, 1, 0) = 1.0; h_coords(1, 1, 1) = 1.0;
    h_coords(1, 2, 0) = 0.0; h_coords(1, 2, 1) = 1.0;
    h_f(1) = 3.0;
    
    // Copy host data to device
    Kokkos::deep_copy(coords, h_coords);
    Kokkos::deep_copy(f, h_f);
    
    // Define material property and create ElementStiffness object
    double kappa = 1.0;
    ElementStiffness stiffness(kappa);
    
    // Compute load vectors
    stiffness.computeTriangleLoadVectorKokkos(coords, f, Fe);
    
    // Copy results back to host for verification
    auto h_Fe = Kokkos::create_mirror_view(Fe);
    Kokkos::deep_copy(h_Fe, Fe);
    
    // Expected load vectors
    double expected1[3] = {2.0/6.0, 2.0/6.0, 2.0/6.0}; // First triangle (area = 0.5, f = 2.0)
    double expected2[3] = {3.0/6.0, 3.0/6.0, 3.0/6.0}; // Second triangle (area = 0.5, f = 3.0)
    
    // Check results for first triangle
    for (int i = 0; i < 3; i++) {
        REQUIRE(h_Fe(0, i) == Catch::Approx(expected1[i]).margin(1e-10));
    }
    
    // Check results for second triangle
    for (int i = 0; i < 3; i++) {
        REQUIRE(h_Fe(1, i) == Catch::Approx(expected2[i]).margin(1e-10));
    }

}