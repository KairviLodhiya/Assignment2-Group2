
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "SparseMatrixCSR.hpp"
#include <Kokkos_Core.hpp>

using Catch::Approx;

TEST_CASE("Identity Matrix Multiply", "[csr]") {

  {
    const int N = 3;
    SparseMatrixCSR A(N, N, N);

    auto h_vals = Kokkos::create_mirror_view(A.values);
    auto h_cols = Kokkos::create_mirror_view(A.col_idx);
    auto h_rows = Kokkos::create_mirror_view(A.row_ptr);

    for (int i = 0; i < N; ++i) {
      h_vals(i) = 1.0;
      h_cols(i) = i;
      h_rows(i) = i;
    }
    h_rows(N) = N;

    Kokkos::deep_copy(A.values, h_vals);
    Kokkos::deep_copy(A.col_idx, h_cols);
    Kokkos::deep_copy(A.row_ptr, h_rows);

    SparseMatrixCSR::ViewVector x("x", N);
    SparseMatrixCSR::ViewVector y("y", N);
    auto h_x = Kokkos::create_mirror_view(x);
    for (int i = 0; i < N; ++i) h_x(i) = i + 1;
    Kokkos::deep_copy(x, h_x);

    A.matvec(x, y);
    auto h_y = Kokkos::create_mirror_view(y);
    Kokkos::deep_copy(h_y, y);

    for (int i = 0; i < N; ++i) {
        REQUIRE(h_y(i) == Approx(h_x(i)));
    }
  }
}
