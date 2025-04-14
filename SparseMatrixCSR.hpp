#ifndef SPARSE_MATRIX_CSR_HPP
#define SPARSE_MATRIX_CSR_HPP

#include <Kokkos_Core.hpp>

class SparseMatrixCSR {
public:
  using exec_space = Kokkos::DefaultExecutionSpace;
  using memory_space = exec_space::memory_space;
  using ViewVector = Kokkos::View<double*, memory_space>;
  using ViewInt = Kokkos::View<int*, memory_space>;

  int numRows, numCols, nnz;
  ViewVector values;
  ViewInt row_ptr;
  ViewInt col_idx;

  SparseMatrixCSR(int rows, int cols, int nnz_estimate);

  void matvec(const ViewVector& x, ViewVector& y) const;
};

#endif
