#include "SparseMatrixCSR.hpp"

SparseMatrixCSR::SparseMatrixCSR(int rows, int cols, int nnz_estimate)
  : numRows(rows), numCols(cols), nnz(nnz_estimate),
    values("values", nnz_estimate),
    row_ptr("row_ptr", rows + 1),
    col_idx("col_idx", nnz_estimate) {}

void SparseMatrixCSR::matvec(const ViewVector& x, ViewVector& y) const {
    // To explicitly capture the class members needed in the lambda
    auto local_values = values;
    auto local_row_ptr = row_ptr;
    auto local_col_idx = col_idx;

    Kokkos::parallel_for("MatVec", Kokkos::RangePolicy<exec_space>(0, numRows), 
    KOKKOS_LAMBDA(const int i) {
      double sum = 0.0;
      for (int idx = local_row_ptr(i); idx < local_row_ptr(i + 1); ++idx) {
        sum += local_values(idx) * x(local_col_idx(idx));
      }
      y(i) = sum;
  });
}
