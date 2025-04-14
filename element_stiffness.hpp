#ifndef ELEMENT_STIFFNESS_HPP
#define ELEMENT_STIFFNESS_HPP

#include <Kokkos_Core.hpp>
// #include "mesh_reader.hpp"

class ElementStiffness {
public:
    /** 
     * @brief Class to compute element stiffness matrix and load vector
     * for triangular and quadrilateral elements.
     * 
     * Parameters: 
     * - kappa: Material property (thermal conductivity)
     * - coords: Coordinates of the element nodes
     * - K: Stiffness matrix
     * - Fe: Load vector
     * - f: Force vector
     * - N: Shape functions
     * - dN_dxi: Shape function derivatives in parametric space
     * - J: Jacobian matrix
     * - Jinv: Inverse Jacobian matrix
     * - detJ: Determinant of Jacobian
     * - dN_dx: Shape function derivatives in physical space
     * - xi, eta: Parametric coordinates
     * - weight: Integration weight
     */

    // Constructor to take material property kappa
    ElementStiffness(double kappa);

    // To compute element stiffness matrix for triangular element
    void computeTriangleStiffnessKokkos(const Kokkos::View<double***>& coords, 
                                      Kokkos::View<double***>& K) const;

    // To compute element stiffness matrix for quadrilateral element
    void computeQuadStiffnessKokkos(const Kokkos::View<double***>& coords, 
                                   Kokkos::View<double***>& K) const;

    // To compute load vector for triangular elements
    void computeTriangleLoadVectorKokkos(const Kokkos::View<double***>& coords, 
                                        const Kokkos::View<double*>& f,
                                        Kokkos::View<double**>& Fe) const;
    
    // To compute load vector for quadrilateral elements
    void computeQuadLoadVectorKokkos(const Kokkos::View<double***>& coords, 
                                     const Kokkos::View<double*>& f,
                                     Kokkos::View<double**>& Fe) const;

    // // testing
    
    // void computeTriangleStiffness(const double coords[3][2], double K[3][3]) const;

    // // for testing
    
    // void computeQuadStiffness(const double coords[4][2], double K[4][4]) const;

    // // for testing
    
    // void computeTriangleLoadVector(const double coords[3][2], const double f, double Fe[3]) const;
    
    // // for testing
    
    // void computeQuadLoadVector(const double coords[4][2], const double f, double Fe[4]) const;

private:
    double kappa_;  // Material property

    // Shape functions for quadrilateral elements
    KOKKOS_INLINE_FUNCTION
    void quadShapeFunctions(const double xi, const double eta, double N[4]) const;

    // Shape function derivatives for quadrilateral elements
    KOKKOS_INLINE_FUNCTION
    void quadShapeFunctionDerivatives(const double xi, const double eta, double dN_dxi[4][2]) const;

    // To calculate Jacobian for quadrilateral elements
    KOKKOS_INLINE_FUNCTION
    void calculateQuadJacobian(const double coords[4][2], const double xi, const double eta, double J[2][2]) const;

    // To calculate Jacobian for triangle elements
    KOKKOS_INLINE_FUNCTION
    void calculateTriangleJacobian(const double coords[3][2], double J[2][2]) const;

    // Calculate inverse and determinant of Jacobian
    KOKKOS_INLINE_FUNCTION
    double calculateInverseJacobian(const double J[2][2], double Jinv[2][2]) const;

    // Calculate shape function derivatives w.r.t. x, y for triangles
    KOKKOS_INLINE_FUNCTION
    void calculateTriangleShapeFunctionDerivatives(const double Jinv[2][2], double dN_dx[3][2]) const;

    // Calculate shape function derivatives w.r.t. x, y for quads
    KOKKOS_INLINE_FUNCTION
    void calculateQuadShapeFunctionDerivatives(const double xi, const double eta, 
                                              const double Jinv[2][2], double dN_dx[4][2]) const;
};

#endif // ELEMENT_STIFFNESS_HPP
