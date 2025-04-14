#include <Kokkos_Core.hpp>
#include <iostream>

class LoadVector {
public:
  using exec_space = Kokkos::DefaultExecutionSpace;
  using memory_space = exec_space::memory_space;
  using ViewVector = Kokkos::View<double*, memory_space>;

  int size;
  ViewVector data;

  LoadVector(int n) : size(n), data("load_vector", n) {}

  
  void zero() {
    Kokkos::parallel_for("ZeroLoadVector", Kokkos::RangePolicy<exec_space>(0, size),
        KOKKOS_LAMBDA(int i) {
                            data(i) = 0.0;
                            });
                }

    void add(int index, double value) {
        Kokkos::parallel_for("AddToLoadVector", 1, KOKKOS_LAMBDA(int) {
          data(index) += value; 
        });
      }
      ViewVector get_data() const {
        return data;
      }
};