#include "element_stiffness.hpp"

ElementStiffness::ElementStiffness(double kappa) : kappa_(kappa) {} // Constructor to initialize kappa

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
            
            // Calculate Jacobian directly from vertices
            double dx_dxi_00 = x2 - x1;  // dx/dxi
            double dx_dxi_01 = x3 - x1;  // dx/deta
            double dx_dxi_10 = y2 - y1;  // dy/dxi
            double dx_dxi_11 = y3 - y1;  // dy/deta
            
            // Calculate determinant and inverse of Jacobian
            double detJ = dx_dxi_00 * dx_dxi_11 - dx_dxi_01 * dx_dxi_10;
            
            // Check for singular Jacobian - use Kokkos::fabs for CUDA compatibility
            if (Kokkos::fabs(detJ) < 1e-10) {
                detJ = 1e-10;
            }
            
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
            dN_dx[0][0] = -1.0 * dxi_dx_00 + -1.0 * dxi_dx_10;  // dN1/dx
            dN_dx[0][1] = -1.0 * dxi_dx_01 + -1.0 * dxi_dx_11;  // dN1/dy
            
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
                        temp += kappa * dN_dx[A][i] * dN_dx[B][i] * detJ;
                    }
                    K(elem, A, B) = temp;
                }
            }
        }
    );
}

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
                    double detJ = dx_dxi_00 * dx_dxi_11 - dx_dxi_01 * dx_dxi_10;
                    
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
                    double N[4];
                    N[0] = 0.25 * (1.0 - xi) * (1.0 - eta);
                    N[1] = 0.25 * (1.0 + xi) * (1.0 - eta);
                    N[2] = 0.25 * (1.0 + xi) * (1.0 + eta);
                    N[3] = 0.25 * (1.0 - xi) * (1.0 + eta);
                    
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

void ElementStiffness::computeQuadLoadVectorKokkos(const Kokkos::View<double***>& coords, 
                                                  const Kokkos::View<double*>& f,
                                                  Kokkos::View<double**>& Fe) const {
    using policy_type = Kokkos::RangePolicy<>;
    const int numElements = coords.extent(0);
    
    Kokkos::parallel_for("ComputeQuadLoadVector", policy_type(0, numElements), 
        KOKKOS_LAMBDA(const int elem) {
            // Initialize Fe to zero for this element
            for (int i = 0; i < 4; i++) {
                Fe(elem, i) = 0.0;
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
                    
                    // Calculate shape function derivatives for Jacobian calculation
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
                    double detJ = dx_dxi_00 * dx_dxi_11 - dx_dxi_01 * dx_dxi_10;
                    
                    // Check for singular Jacobian
                    if (Kokkos::fabs(detJ) < 1e-10) {
                        detJ = 1e-10;
                    }
                    
                    // Calculate shape functions at this point
                    double N[4];
                    N[0] = 0.25 * (1.0 - xi) * (1.0 - eta);
                    N[1] = 0.25 * (1.0 + xi) * (1.0 - eta);
                    N[2] = 0.25 * (1.0 + xi) * (1.0 + eta);
                    N[3] = 0.25 * (1.0 - xi) * (1.0 + eta);
                    
                    // Add contribution to load vector
                    for (int A = 0; A < 4; A++) {
                        localFe[A] += f(elem) * N[A] * detJ * weight;
                    }
                }
            }
            
            // Copy back to Kokkos View
            for (int i = 0; i < 4; i++) {
                Fe(elem, i) = localFe[i];
            }
        }
    );
}

KOKKOS_INLINE_FUNCTION
void ElementStiffness::quadShapeFunctions(const double xi, const double eta, double N[4]) const {
    N[0] = 0.25 * (1.0 - xi) * (1.0 - eta);
    N[1] = 0.25 * (1.0 + xi) * (1.0 - eta);
    N[2] = 0.25 * (1.0 + xi) * (1.0 + eta);
    N[3] = 0.25 * (1.0 - xi) * (1.0 + eta);
}

KOKKOS_INLINE_FUNCTION
void ElementStiffness::quadShapeFunctionDerivatives(const double xi, const double eta, double dN_dxi[4][2]) const {
    // dN/dxi
    dN_dxi[0][0] = -0.25 * (1.0 - eta);
    dN_dxi[1][0] =  0.25 * (1.0 - eta);
    dN_dxi[2][0] =  0.25 * (1.0 + eta);
    dN_dxi[3][0] = -0.25 * (1.0 + eta);
    
    // dN/deta
    dN_dxi[0][1] = -0.25 * (1.0 - xi);
    dN_dxi[1][1] = -0.25 * (1.0 + xi);
    dN_dxi[2][1] =  0.25 * (1.0 + xi);
    dN_dxi[3][1] =  0.25 * (1.0 - xi);
}

KOKKOS_INLINE_FUNCTION
void ElementStiffness::calculateTriangleJacobian(const double coords[3][2], double J[2][2]) const {
    // For triangles:
    // x = N1*x1 + N2*x2 + N3*x3 = (1-xi-eta)*x1 + xi*x2 + eta*x3
    // y = N1*y1 + N2*y2 + N3*y3 = (1-xi-eta)*y1 + xi*y2 + eta*y3
    
    J[0][0] = coords[1][0] - coords[0][0];  // dx/dxi
    J[0][1] = coords[2][0] - coords[0][0];  // dx/deta
    J[1][0] = coords[1][1] - coords[0][1];  // dy/dxi
    J[1][1] = coords[2][1] - coords[0][1];  // dy/deta
}

KOKKOS_INLINE_FUNCTION
void ElementStiffness::calculateQuadJacobian(const double coords[4][2], const double xi, const double eta, double J[2][2]) const {
    // Initialize Jacobian to zero
    J[0][0] = 0.0; J[0][1] = 0.0;
    J[1][0] = 0.0; J[1][1] = 0.0;
    
    // Calculate shape function derivatives w.r.t. xi, eta
    double dN_dxi[4][2];
    quadShapeFunctionDerivatives(xi, eta, dN_dxi);
    
    // Calculate Jacobian
    for (int i = 0; i < 4; i++) {
        J[0][0] += dN_dxi[i][0] * coords[i][0];  // dx/dxi
        J[0][1] += dN_dxi[i][1] * coords[i][0];  // dx/deta
        J[1][0] += dN_dxi[i][0] * coords[i][1];  // dy/dxi
        J[1][1] += dN_dxi[i][1] * coords[i][1];  // dy/deta
    }
}

KOKKOS_INLINE_FUNCTION
double ElementStiffness::calculateInverseJacobian(const double J[2][2], double Jinv[2][2]) const {
    double detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0];
    
    // Check for singular Jacobian
    if (fabs(detJ) < 1e-10) {  // fabs is used to avoid division by zero to avoid NaN
        detJ = 1e-10;
    }
    
    double invDetJ = 1.0 / detJ;
    Jinv[0][0] =  J[1][1] * invDetJ;
    Jinv[0][1] = -J[0][1] * invDetJ;
    Jinv[1][0] = -J[1][0] * invDetJ;
    Jinv[1][1] =  J[0][0] * invDetJ;
    
    return detJ;
}

KOKKOS_INLINE_FUNCTION
void ElementStiffness::calculateTriangleShapeFunctionDerivatives(const double Jinv[2][2], double dN_dx[3][2]) const {
    // For triangles:
    // dN1/dxi = -1, dN1/deta = -1
    // dN2/dxi =  1, dN2/deta =  0
    // dN3/dxi =  0, dN3/deta =  1
    
    // Apply chain rule: dN/dx = dN/dxi * dxi/dx + dN/deta * deta/dx
    double dN_dxi[3][2] = {
        {-1.0, -1.0},  // dN1/dxi, dN1/deta
        { 1.0,  0.0},  // dN2/dxi, dN2/deta
        { 0.0,  1.0}   // dN3/dxi, dN3/deta
    };
    
    for (int A = 0; A < 3; A++) {
        dN_dx[A][0] = dN_dxi[A][0] * Jinv[0][0] + dN_dxi[A][1] * Jinv[1][0];  // dN/dx
        dN_dx[A][1] = dN_dxi[A][0] * Jinv[0][1] + dN_dxi[A][1] * Jinv[1][1];  // dN/dy
    }
}

KOKKOS_INLINE_FUNCTION
void ElementStiffness::calculateQuadShapeFunctionDerivatives(const double xi, const double eta, 
                                           const double Jinv[2][2], double dN_dx[4][2]) const {
    // Calculate shape function derivatives w.r.t. xi, eta
    double dN_dxi[4][2];
    quadShapeFunctionDerivatives(xi, eta, dN_dxi);
    
    // Apply chain rule to get derivatives w.r.t. x, y
    for (int A = 0; A < 4; A++) {
        dN_dx[A][0] = dN_dxi[A][0] * Jinv[0][0] + dN_dxi[A][1] * Jinv[1][0];  // dN/dx
        dN_dx[A][1] = dN_dxi[A][0] * Jinv[0][1] + dN_dxi[A][1] * Jinv[1][1];  // dN/dy
    }
}

// // Implementation that does not use Kokkos (for testing)
// void ElementStiffness::computeTriangleStiffness(const double coords[3][2], double K[3][3]) const {
//     // Calculate Jacobian
//     double dx_dxi[2][2];  // Jacobian matrix
//     calculateTriangleJacobian(coords, dx_dxi);
    
//     // Calculate inverse and determinant of Jacobian
//     double dxi_dx[2][2];  // Inverse Jacobian
//     double detJ = calculateInverseJacobian(dx_dxi, dxi_dx);
    
//     // Calculate derivatives of shape functions w.r.t. x,y
//     double dN_dx[3][2];
//     calculateTriangleShapeFunctionDerivatives(dxi_dx, dN_dx);
    
//     // Calculate stiffness matrix
//     for (int A = 0; A < 3; A++) {
//         for (int B = 0; B < 3; B++) {
//             K[A][B] = 0.0;
//             for (int i = 0; i < 2; i++) {
//                 K[A][B] += kappa_ * dN_dx[A][i] * dN_dx[B][i] * detJ;
//             }
//         }
//     }
// }


// void ElementStiffness::computeQuadStiffness(const double coords[4][2], double K[4][4]) const {
//     // Initialize stiffness matrix to zero
//     for (int i = 0; i < 4; i++) {
//         for (int j = 0; j < 4; j++) {
//             K[i][j] = 0.0;
//         }
//     }
    
//     // Gauss quadrature points and weights for 2x2 quadrature
//     const double gaussPoints[2] = {-1.0/sqrt(3.0), 1.0/sqrt(3.0)};
//     const double gaussWeights[2] = {1.0, 1.0};
    
//     // Loop over quadrature points
//     for (int i = 0; i < 2; i++) {
//         for (int j = 0; j < 2; j++) {
//             double xi = gaussPoints[i];
//             double eta = gaussPoints[j];
//             double weight = gaussWeights[i] * gaussWeights[j];
            
//             // Calculate Jacobian at this integration point
//             double dx_dxi[2][2];
//             calculateQuadJacobian(coords, xi, eta, dx_dxi);
            
//             // Calculate inverse and determinant of Jacobian
//             double dxi_dx[2][2];
//             double detJ = calculateInverseJacobian(dx_dxi, dxi_dx);
            
//             // Calculate derivatives of shape functions w.r.t. x,y
//             double dN_dx[4][2];
//             calculateQuadShapeFunctionDerivatives(xi, eta, dxi_dx, dN_dx);
            
//             // Add contribution to stiffness matrix
//             for (int A = 0; A < 4; A++) {
//                 for (int B = 0; B < 4; B++) {
//                     for (int k = 0; k < 2; k++) {
//                         K[A][B] += kappa_ * dN_dx[A][k] * dN_dx[B][k] * detJ * weight;
//                     }
//                 }
//             }
//         }
//     }
// }


// void ElementStiffness::computeTriangleLoadVector(const double coords[3][2], const double f, double Fe[3]) const {
//     // for the triangle element gauss quadrature is not needed using the exact integration method
//     // Calculate Jacobian
//     double dx_dxi[2][2];
//     calculateTriangleJacobian(coords, dx_dxi);
    
//     // Calculate determinant of Jacobian
//     double detJ = dx_dxi[0][0] * dx_dxi[1][1] - dx_dxi[0][1] * dx_dxi[1][0];
    
//     // Area of triangle = detJ/2
//     double area = detJ * 0.5;
    
//     // Shape functions at centroid (1/3, 1/3)
//     Fe[0] = f * area / 3.0;
//     Fe[1] = f * area / 3.0;
//     Fe[2] = f * area / 3.0;
// }

// void ElementStiffness::computeQuadLoadVector(const double coords[4][2], const double f, double Fe[4]) const {
//     // Initialize load vector to zero
//     for (int i = 0; i < 4; i++) {
//         Fe[i] = 0.0;
//     }
    
//     // Gauss quadrature points and weights for 2x2 quadrature
//     const double gaussPoints[2] = {-1.0/sqrt(3.0), 1.0/sqrt(3.0)};
//     const double gaussWeights[2] = {1.0, 1.0};
    
//     // Loop over quadrature points
//     for (int i = 0; i < 2; i++) {
//         for (int j = 0; j < 2; j++) {
//             double xi = gaussPoints[i];
//             double eta = gaussPoints[j];
//             double weight = gaussWeights[i] * gaussWeights[j];
            
//             // Calculate Jacobian at this integration point
//             double dx_dxi[2][2];
//             calculateQuadJacobian(coords, xi, eta, dx_dxi);
            
//             // Calculate determinant of Jacobian
//             double detJ = dx_dxi[0][0] * dx_dxi[1][1] - dx_dxi[0][1] * dx_dxi[1][0];
            
//             // Calculate shape functions at this point
//             double N[4];
//             quadShapeFunctions(xi, eta, N);
            
//             // Add contribution to load vector
//             for (int A = 0; A < 4; A++) {
//                 Fe[A] += f * N[A] * detJ * weight;
//             }
//         }
//     }
// }