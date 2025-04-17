#include "element_stiffness.hpp"

ElementStiffness::ElementStiffness(double kappa) : kappa_(kappa) {} // Constructor to initialize kappa

// Compute stiffness matrix for triangular elements
// The stiffness matrix is computed using the shape functions and their derivatives 
// Transformation from parametric to physical coordinates is done using the Jacobian
void ElementStiffness::computeTriangleStiffnessKokkos(const Kokkos::View<double***>& coords, 
                                                     Kokkos::View<double***>& K) const { // Compute stiffness matrix for triangle elements
    using policy_type = Kokkos::RangePolicy<>;  // Rangepolicy used to parallelize the loop
    const int numElements = coords.extent(0);   // Get number of elements using extent of the Kokkos view
    const double kappa = kappa_; // Capture kappa_ value for lambda
    
    Kokkos::parallel_for("ComputeTriangleStiffness", policy_type(0, numElements),  // Parallel for loop from 0 to numElements
        KOKKOS_LAMBDA(const int elem) {
            // Use direct Kokkos::View instead of local arrays to improve CUDA compatibility
            double x1 = coords(elem, 0, 0);
            double y1 = coords(elem, 0, 1);
            double x2 = coords(elem, 1, 0);
            double y2 = coords(elem, 1, 1);
            double x3 = coords(elem, 2, 0);
            double y3 = coords(elem, 2, 1);

            // the global coordinates of the interial points is given by 
            // x = N1*x1 + N2*x2 + N3*x3 and y = N1*y1 + N2*y2 + N3*y3
            // where N1, N2, N3 are the shape functions
            // For triangles:
            // N1 = 1 - xi - eta
            // N2 = xi
            // N3 = eta

            // the matrix for the Jacobian is given by
            // j = [ dx/dxi  dx/deta;
            //       dy/dxi  dy/deta]
            // where dx/dxi = x2 - x1, 
            // dx/deta = x3 - x1,
            // dy/dxi = y2 - y1, 
            // dy/deta = y3 - y1
            // Calculate Jacobian directly from vertices
            double dx_dxi_00 = x2 - x1;  // dx/dxi
            double dx_dxi_01 = x3 - x1;  // dx/deta
            double dx_dxi_10 = y2 - y1;  // dy/dxi
            double dx_dxi_11 = y3 - y1;  // dy/deta
            
            // Calculate determinant and inverse of Jacobian
            //double detJ = dx_dxi_00 * dx_dxi_11 - dx_dxi_01 * dx_dxi_10;
            double detJ = Kokkos::fabs(dx_dxi_00 * dx_dxi_11 - dx_dxi_01 * dx_dxi_10);
            
            // Check for singular Jacobian - use Kokkos::fabs for CUDA compatibility
            if (Kokkos::fabs(detJ) < 1e-10) {
                detJ = 1e-10;
            }
            
            // Calculate inverse of Jacobian
            // J^-1 = 1/detJ * [ dy/deta  -dx/deta;
            //                  -dy/dxi  dx/dxi]

            double invDetJ = 1.0 / detJ;
            double dxi_dx_00 =  dx_dxi_11 * invDetJ;
            double dxi_dx_01 = -dx_dxi_01 * invDetJ;
            double dxi_dx_10 = -dx_dxi_10 * invDetJ;
            double dxi_dx_11 =  dx_dxi_00 * invDetJ;
            
            // Calculate derivatives of shape functions w.r.t. x,y
            // For triangles:
            // dN1/dxi = -1, dN1/deta = -1
            // dN2/dxi =  1, dN2/deta =  0
            // dN3/dxi =  0, dN3/deta =  1
            
            // Apply chain rule: dN/dx = dN/dxi * dxi/dx + dN/deta * deta/dx
            double dN_dx[3][2];
            
            // Node 1
            dN_dx[0][0] = -1.0 * dxi_dx_00 -1.0 * dxi_dx_10;  // dN1/dx
            dN_dx[0][1] = -1.0 * dxi_dx_01 -1.0 * dxi_dx_11;  // dN1/dy
            
            // Node 2
            dN_dx[1][0] = 1.0 * dxi_dx_00 + 0.0 * dxi_dx_10;  // dN2/dx
            dN_dx[1][1] = 1.0 * dxi_dx_01 + 0.0 * dxi_dx_11;  // dN2/dy
            
            // Node 3
            dN_dx[2][0] = 0.0 * dxi_dx_00 + 1.0 * dxi_dx_10;  // dN3/dx
            dN_dx[2][1] = 0.0 * dxi_dx_01 + 1.0 * dxi_dx_11;  // dN3/dy
            
            // Calculate stiffness matrix
            for (int A = 0; A < 3; A++) {
                for (int B = 0; B < 3; B++) {
                    double temp = 0.0;
                    for (int i = 0; i < 2; i++) {
                        temp += kappa * detJ * dN_dx[A][i] * dN_dx[B][i] ;
                    }
                    K(elem, A, B) = temp;
                }
            }
        }
    );
}

// Compute stiffness matrix for quadrilateral elements
// The stiffness matrix is computed using the shape functions and their derivatives
// Transformation from parametric to physical coordinates is done using the Jacobian
void ElementStiffness::computeQuadStiffnessKokkos(const Kokkos::View<double***>& coords, 
                                                 Kokkos::View<double***>& K) const {  // view used because it is a 3D array
    using policy_type = Kokkos::RangePolicy<>;
    const int numElements = coords.extent(0);
    const double kappa = kappa_; // Capture kappa_ value for lambda
    
    Kokkos::parallel_for("ComputeQuadStiffness", policy_type(0, numElements), 
        KOKKOS_LAMBDA(const int elem) {
            // Initialize K to zero for this element
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    K(elem, i, j) = 0.0;
                }
            }
            
            // Gauss quadrature points and weights for 2x2 quadrature
            const double gaussPoints[2] = {-1.0/Kokkos::sqrt(3.0), 1.0/Kokkos::sqrt(3.0)};
            const double gaussWeights[2] = {1.0, 1.0};
            
            // Loop over quadrature points
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    double xi = gaussPoints[i];
                    double eta = gaussPoints[j];
                    double weight = gaussWeights[i] * gaussWeights[j];
                    
                    // Calculate shape function derivatives w.r.t. xi, eta
                    double dN_dxi[4];
                    double dN_deta[4];
                    
                    // dN/dxi
                    dN_dxi[0] = -0.25 * (1.0 - eta);
                    dN_dxi[1] =  0.25 * (1.0 - eta);
                    dN_dxi[2] =  0.25 * (1.0 + eta);
                    dN_dxi[3] = -0.25 * (1.0 + eta);
                    
                    // dN/deta
                    dN_deta[0] = -0.25 * (1.0 - xi);
                    dN_deta[1] = -0.25 * (1.0 + xi);
                    dN_deta[2] =  0.25 * (1.0 + xi);
                    dN_deta[3] =  0.25 * (1.0 - xi);
                    
                    // Calculate Jacobian at this integration point
                    double dx_dxi_00 = 0.0;
                    double dx_dxi_01 = 0.0;
                    double dx_dxi_10 = 0.0;
                    double dx_dxi_11 = 0.0;
                    
                    // Calculate Jacobian
                    for (int n = 0; n < 4; n++) {
                        dx_dxi_00 += dN_dxi[n] * coords(elem, n, 0);   // dx/dxi
                        dx_dxi_01 += dN_deta[n] * coords(elem, n, 0);  // dx/deta
                        dx_dxi_10 += dN_dxi[n] * coords(elem, n, 1);   // dy/dxi
                        dx_dxi_11 += dN_deta[n] * coords(elem, n, 1);  // dy/deta
                    }
                    
                    // Calculate determinant of Jacobian
                    double detJ = Kokkos::fabs(dx_dxi_00 * dx_dxi_11 - dx_dxi_01 * dx_dxi_10);
                    
                    // Check for singular Jacobian
                    if (Kokkos::fabs(detJ) < 1e-10) {
                        detJ = 1e-10;
                    }
                    
                    // Calculate inverse of Jacobian
                    double invDetJ = 1.0 / detJ;
                    double dxi_dx_00 =  dx_dxi_11 * invDetJ;
                    double dxi_dx_01 = -dx_dxi_01 * invDetJ;
                    double dxi_dx_10 = -dx_dxi_10 * invDetJ;
                    double dxi_dx_11 =  dx_dxi_00 * invDetJ;
                    
                    // Calculate derivatives of shape functions w.r.t. x,y
                    double dN_dx[4][2];
                    
                    // Apply chain rule to get derivatives w.r.t. x, y
                    for (int A = 0; A < 4; A++) {
                        dN_dx[A][0] = dN_dxi[A] * dxi_dx_00 + dN_deta[A] * dxi_dx_10;  // dN/dx
                        dN_dx[A][1] = dN_dxi[A] * dxi_dx_01 + dN_deta[A] * dxi_dx_11;  // dN/dy
                    }
                    
                    // Calculate shape functions at this point (needed for load vector computations)
                    // double N[4];
                    // N[0] = 0.25 * (1.0 - xi) * (1.0 - eta);
                    // N[1] = 0.25 * (1.0 + xi) * (1.0 - eta);
                    // N[2] = 0.25 * (1.0 + xi) * (1.0 + eta);
                    // N[3] = 0.25 * (1.0 - xi) * (1.0 + eta);
                    
                    // Add contribution to stiffness matrix
                    for (int A = 0; A < 4; A++) {
                        for (int B = 0; B < 4; B++) {
                            double temp = 0.0;
                            for (int k = 0; k < 2; k++) {
                                temp += kappa * dN_dx[A][k] * dN_dx[B][k] * detJ * weight;
                            }
                            K(elem, A, B) += temp;
                        }
                    }
                }
            }
        }
    );
}

// Function to create the load vector for triangular elements from the force vector
// The Load vector is transforming the force vector from the global coordinates to the local coordinates
void ElementStiffness::computeTriangleLoadVectorKokkos(const Kokkos::View<double***>& coords, 
                                                      const Kokkos::View<double*>& f,
                                                      Kokkos::View<double**>& Fe) const {
    using policy_type = Kokkos::RangePolicy<>;
    const int numElements = coords.extent(0);
    
    Kokkos::parallel_for("ComputeTriangleLoadVector", policy_type(0, numElements), 
        KOKKOS_LAMBDA(const int elem) {
            // Get coordinates directly
            double x1 = coords(elem, 0, 0);
            double y1 = coords(elem, 0, 1);
            double x2 = coords(elem, 1, 0);
            double y2 = coords(elem, 1, 1);
            double x3 = coords(elem, 2, 0);
            double y3 = coords(elem, 2, 1);
            
            // Calculate Jacobian
            double dx_dxi_00 = x2 - x1;  // dx/dxi 
            double dx_dxi_01 = x3 - x1;  // dx/deta
            double dx_dxi_10 = y2 - y1;  // dy/dxi
            double dx_dxi_11 = y3 - y1;  // dy/deta
            
            // Calculate determinant of Jacobian
            double detJ = dx_dxi_00 * dx_dxi_11 - dx_dxi_01 * dx_dxi_10;
            
            // Area of triangle = detJ/2
            double area = Kokkos::fabs(detJ) * 0.5;
            
            // Load vector - equal distribution to all nodes (1/3 each)
            double nodeValue = f(elem) * area / 3.0;
            Fe(elem, 0) = nodeValue;
            Fe(elem, 1) = nodeValue;
            Fe(elem, 2) = nodeValue;
        }
    );
}

// Function to create the load vector for quadrilateral elements from the force vector
// The Load vector is transforming the force vector from the global coordinates to the local coordinates
void ElementStiffness::computeQuadLoadVectorKokkos(
    const Kokkos::View<double***>& coords, 
    const Kokkos::View<double*>& f,
    Kokkos::View<double**>& Fe) const 
{
    using policy_type = Kokkos::RangePolicy<>;
    const int numElements = coords.extent(0);

    Kokkos::parallel_for("ComputeQuadLoadVector", policy_type(0, numElements),
        KOKKOS_LAMBDA(const int elem) {
            // Local load vector
            double localFe[4] = {0.0, 0.0, 0.0, 0.0};

            // Gauss quadrature points and weights for 2x2 integration
            const double gaussPoints[2] = { -1.0 / Kokkos::sqrt(3.0), 1.0 / Kokkos::sqrt(3.0) };
            const double gaussWeights[2] = { 1.0, 1.0 };

            // Loop over Gauss points
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    double xi = gaussPoints[i];
                    double eta = gaussPoints[j];
                    double weight = gaussWeights[i] * gaussWeights[j];

                    // Derivatives of shape functions w.r.t xi and eta
                    double dN_dxi[4] = {
                        -0.25 * (1.0 - eta),
                         0.25 * (1.0 - eta),
                         0.25 * (1.0 + eta),
                        -0.25 * (1.0 + eta)
                    };

                    double dN_deta[4] = {
                        -0.25 * (1.0 - xi),
                        -0.25 * (1.0 + xi),
                         0.25 * (1.0 + xi),
                         0.25 * (1.0 - xi)
                    };

                    // Compute Jacobian components
                    double dx_dxi_00 = 0.0, dx_dxi_01 = 0.0;
                    double dx_dxi_10 = 0.0, dx_dxi_11 = 0.0;

                    for (int n = 0; n < 4; ++n) {
                        dx_dxi_00 += dN_dxi[n] * coords(elem, n, 0); // dx/dxi
                        dx_dxi_01 += dN_deta[n] * coords(elem, n, 0); // dx/deta
                        dx_dxi_10 += dN_dxi[n] * coords(elem, n, 1); // dy/dxi
                        dx_dxi_11 += dN_deta[n] * coords(elem, n, 1); // dy/deta
                    }

                    // Determinant of Jacobian
                    double detJ = dx_dxi_00 * dx_dxi_11 - dx_dxi_01 * dx_dxi_10;
                    if (Kokkos::fabs(detJ) < 1e-10) {
                        detJ = 1e-10; // Avoid division by zero or near-singular matrix
                    }

                    // Shape functions at Gauss point
                    double N[4] = {
                        0.25 * (1.0 - xi) * (1.0 - eta),
                        0.25 * (1.0 + xi) * (1.0 - eta),
                        0.25 * (1.0 + xi) * (1.0 + eta),
                        0.25 * (1.0 - xi) * (1.0 + eta)
                    };

                    // Assemble local load vector
                    for (int A = 0; A < 4; ++A) {
                        localFe[A] += f(elem) * N[A] * detJ * weight;
                    }
                }
            }

            // Store local result in global view
            for (int i = 0; i < 4; ++i) {
                Fe(elem, i) = localFe[i];
            }
        }
    );
}
