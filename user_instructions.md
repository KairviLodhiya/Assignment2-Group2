# Assignment 2: Parallel Finite Element Solver

This project implements a parallel 2D Finite Element Method (FEM) solver using [Kokkos](https://github.com/kokkos/kokkos) for performance portability and [Catch2](https://github.com/catchorg/Catch2) for unit testing. It includes functionality for:

- Reading GMSH `.msh` mesh files
- Computing local element stiffness matrices and load vectors
- Assembling the global system (CSR sparse matrix + load vector)
- Running test cases for validation

---

## Dependencies

Make sure the following are installed and available:

- **CMake** ≥ 3.20
- **C++20-compatible compiler** (e.g. `g++`, `clang++`)
- [**Kokkos**](https://github.com/kokkos/kokkos)
- [**Catch2**](https://github.com/catchorg/Catch2)

You can install Kokkos and Catch2 manually or using a package manager like `vcpkg`, or build them from source and install to a custom prefix.

---

## Directory Structure

. ├── CMakeLists.txt ├── mesh_reader_kokkos.cpp / .hpp ├── element_stiffness.cpp / .hpp ├── SparseMatrixCSR.cpp / .hpp ├── LoadVector.cpp / .hpp ├── coo_to_csr.hpp ├── assemble_system.hpp ├── test_main.cpp ├── test_*.cpp ├── README.md

yaml
Copy
Edit

---

## Running the Code

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd assignment2
2. Configure and Build
bash
Copy
Edit
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j
If using a custom install prefix for Kokkos or Catch2, add:

bash
Copy
Edit
cmake .. -DCMAKE_PREFIX_PATH="/path/to/kokkos;/path/to/catch2"
3. Run All Tests
bash
Copy
Edit
./all_tests
