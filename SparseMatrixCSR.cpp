#include "SparseMatrixCSR.hpp"


SparseMatrixCSR::SparseMatrixCSR(int rows, int cols, int nnz_estimate)
  : numRows(rows), numCols(cols), nnz(nnz_estimate),
    values("values", nnz_estimate),
    row_ptr("row_ptr", rows + 1),
    col_idx("col_idx", nnz_estimate) {}

    void SparseMatrixCSR::matvec(const ViewVector& x, ViewVector& y) const {
        Kokkos::parallel_for("MatVec", Kokkos::RangePolicy<exec_space>(0, numRows), KOKKOS_LAMBDA(const int i) {
          double sum = 0.0;
          for (int idx = row_ptr(i); idx < row_ptr(i + 1); ++idx) {
            sum += values(idx) * x(col_idx(idx));
          }
          y(i) = sum;
        });
}
