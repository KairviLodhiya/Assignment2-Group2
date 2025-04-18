// === coo_to_csr_kokkos.hpp ===
#ifndef COO_TO_CSR_KOKKOS_HPP
#define COO_TO_CSR_KOKKOS_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include "../Global/SparseMatrixCSR.hpp"

struct COOMatrixDevice {
    int rows, cols;
    Kokkos::View<int*> row_idx;
    Kokkos::View<int*> col_idx;
    Kokkos::View<double*> values;
};

inline COOMatrixDevice create_coo_matrix_device(int rows, int cols, int nnz_estimate) {
    return {
        rows,
        cols,
        Kokkos::View<int*>("coo_rows", nnz_estimate),
        Kokkos::View<int*>("coo_cols", nnz_estimate),
        Kokkos::View<double*>("coo_vals", nnz_estimate)
    };
}

inline void coo_to_csr_kokkos(const COOMatrixDevice& coo, SparseMatrixCSR& csr) {
    int rows = coo.rows;
    int nnz = coo.values.extent(0);

    csr.numRows = rows;
    csr.numCols = coo.cols;
    csr.nnz = nnz;
    csr.values = Kokkos::View<double*>("csr_values", nnz);
    csr.col_idx = Kokkos::View<int*>("csr_cols", nnz);
    csr.row_ptr = Kokkos::View<int*>("csr_row_ptr", rows + 1);

    // Step 1: Count number of nonzeros per row
    Kokkos::parallel_for("CountNNZ", nnz, KOKKOS_LAMBDA(const int k) {
        Kokkos::atomic_inc(&csr.row_ptr(coo.row_idx(k) + 1));
    });

    // Step 2: Exclusive scan on row_ptr
    Kokkos::parallel_scan("ExclusiveScan", rows + 1, KOKKOS_LAMBDA(const int i, int& update, const bool final) {
        int val = csr.row_ptr(i);
        if (final) csr.row_ptr(i) = update;
        update += val;
    });

    // Step 3: Fill col_idx and values using row_ptr as scratch space
    Kokkos::View<int*> row_offset("row_offset", rows);
    Kokkos::parallel_for("InitOffset", rows, KOKKOS_LAMBDA(const int i) {
        row_offset(i) = csr.row_ptr(i);
    });

    Kokkos::parallel_for("ScatterEntries", nnz, KOKKOS_LAMBDA(const int k) {
        int i = coo.row_idx(k);
        int insert_pos = Kokkos::atomic_fetch_add(&row_offset(i), 1);
        csr.col_idx(insert_pos) = coo.col_idx(k);
        csr.values(insert_pos) = coo.values(k);
    });
}

#endif // COO_TO_CSR_KOKKOS_HPP