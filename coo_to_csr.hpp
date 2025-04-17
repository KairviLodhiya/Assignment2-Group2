#ifndef COO_TO_CSR_HPP
#define COO_TO_CSR_HPP

#include <vector>
#include <tuple>
#include <algorithm>
#include <stdexcept>

struct COOMatrix {
    int rows, cols;
    std::vector<int> row_idx;
    std::vector<int> col_idx;
    std::vector<double> values;
};

inline COOMatrix create_coo_matrix(int rows, int cols) {
    return COOMatrix{rows, cols, {}, {}, {}};
}

inline void add_coo_entry(COOMatrix& coo, int i, int j, double v) {
    coo.row_idx.push_back(i);
    coo.col_idx.push_back(j);
    coo.values.push_back(v);
}

inline void coo_to_csr(const COOMatrix& coo, SparseMatrixCSR& csr) {
    if (coo.row_idx.size() != coo.col_idx.size() || coo.row_idx.size() != coo.values.size()) {
        throw std::runtime_error("Invalid COO matrix");
    }

    const int nnz = coo.values.size();
    csr.nnz = nnz;
    csr.numRows = coo.rows;
    csr.numCols = coo.cols;
    csr.values = Kokkos::View<double*>("csr_values", nnz);
    csr.col_idx = Kokkos::View<int*>("csr_cols", nnz);
    csr.row_ptr = Kokkos::View<int*>("csr_row_ptr", coo.rows + 1);

    std::vector<std::tuple<int, int, double>> triplets;
    for (size_t k = 0; k < coo.values.size(); ++k) {
        triplets.emplace_back(coo.row_idx[k], coo.col_idx[k], coo.values[k]);
    }
    std::sort(triplets.begin(), triplets.end());

    std::vector<int> row_ptr(coo.rows + 1, 0);
    std::vector<int> col_idx(nnz);
    std::vector<double> values(nnz);

    for (const auto& [i, j, v] : triplets) row_ptr[i + 1]++;
    for (int i = 1; i <= coo.rows; ++i) row_ptr[i] += row_ptr[i - 1];

    for (size_t k = 0; k < triplets.size(); ++k) {
        const auto& [i, j, v] = triplets[k];
        col_idx[k] = j;
        values[k] = v;
    }

    Kokkos::deep_copy(csr.col_idx, Kokkos::View<int*, Kokkos::HostSpace>(col_idx.data(), nnz));
    Kokkos::deep_copy(csr.values, Kokkos::View<double*, Kokkos::HostSpace>(values.data(), nnz));
    Kokkos::deep_copy(csr.row_ptr, Kokkos::View<int*, Kokkos::HostSpace>(row_ptr.data(), coo.rows + 1));
}

#endif // COO_TO_CSR_HPP