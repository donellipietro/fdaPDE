// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include <fdaPDE/core/utils/Symbols.h>
#include <fdaPDE/models/regression/SRPDE.h>
using fdaPDE::models::SRPDE;
#include <fdaPDE/core/utils/DataStructures/BlockFrame.h>
#include <fdaPDE/models/ModelTraits.h>
#include <fdaPDE/core/FEM/PDE.h>
using fdaPDE::core::FEM::DefaultOperator;
using fdaPDE::core::FEM::PDE;
#include <fdaPDE/core/FEM/operators/SpaceVaryingFunctors.h>
using fdaPDE::core::FEM::SpaceVaryingAdvection;
using fdaPDE::core::FEM::SpaceVaryingDiffusion;
using fdaPDE::core::FEM::SpaceVaryingReaction;
#include <fdaPDE/core/MESH/Mesh.h>
using fdaPDE::core::MESH::Mesh;
#include <fdaPDE/models/SamplingDesign.h>

#include "Common.h"

// expose RegularizingPDE as possible argument to other Rcpp modules
RCPP_EXPOSED_AS  (Laplacian_2D_Order1)
RCPP_EXPOSED_WRAP(Laplacian_2D_Order1)

RCPP_MODULE(Laplacian_2D_Order1) {
  Rcpp::class_<Laplacian_2D_Order1>("Laplacian_2D_Order1")
    .constructor<Rcpp::List>()
    // getters
    .method("get_quadrature_nodes", &Laplacian_2D_Order1::get_quadrature_nodes)
    .method("get_dofs_coordinates", &Laplacian_2D_Order1::get_dofs_coordinates)
    // setters
    .method("set_dirichlet_bc",     &Laplacian_2D_Order1::set_dirichlet_bc)
    .method("R0", &Laplacian_2D_Order1::R0)
    .method("init", &Laplacian_2D_Order1::init)
    .method("set_forcing_term",     &Laplacian_2D_Order1::set_forcing_term);
}
// expose RegularizingPDE as possible argument to other Rcpp modules
RCPP_EXPOSED_AS  (ConstantCoefficients_2D_Order1)
RCPP_EXPOSED_WRAP(ConstantCoefficients_2D_Order1)

RCPP_MODULE(ConstantCoefficients_2D_Order1) {
  Rcpp::class_<ConstantCoefficients_2D_Order1>("ConstantCoefficients_2D_Order1")
    .constructor<Rcpp::List>()
    // getters
    .method("get_quadrature_nodes", &ConstantCoefficients_2D_Order1::get_quadrature_nodes)
    .method("get_dofs_coordinates", &ConstantCoefficients_2D_Order1::get_dofs_coordinates)
    // setters
    .method("set_dirichlet_bc",     &ConstantCoefficients_2D_Order1::set_dirichlet_bc)
    .method("set_forcing_term",     &ConstantCoefficients_2D_Order1::set_forcing_term)
    .method("set_PDE_parameters",   &ConstantCoefficients_2D_Order1::set_PDE_parameters);
}
// expose RegularizingPDE as possible argument to other Rcpp modules
RCPP_EXPOSED_AS  (SpaceVarying_2D_Order1)
RCPP_EXPOSED_WRAP(SpaceVarying_2D_Order1)

RCPP_MODULE(SpaceVarying_2D_Order1) {
  Rcpp::class_<SpaceVarying_2D_Order1>("SpaceVarying_2D_Order1")
    .constructor<Rcpp::List>()
    // getters
    .method("get_quadrature_nodes", &SpaceVarying_2D_Order1::get_quadrature_nodes)
    .method("get_dofs_coordinates", &SpaceVarying_2D_Order1::get_dofs_coordinates)
    // setters
    .method("set_dirichlet_bc",     &SpaceVarying_2D_Order1::set_dirichlet_bc)
    .method("set_forcing_term",     &SpaceVarying_2D_Order1::set_forcing_term)
    .method("set_PDE_parameters",   &SpaceVarying_2D_Order1::set_PDE_parameters);
}


RCPP_EXPOSED_AS  (Laplacian_3D_Order1)
RCPP_EXPOSED_WRAP(Laplacian_3D_Order1)

RCPP_MODULE(Laplacian_3D_Order1) {
  Rcpp::class_<Laplacian_3D_Order1>("Laplacian_3D_Order1")
    .constructor<Rcpp::List>()
    // getters
    .method("get_quadrature_nodes", &Laplacian_3D_Order1::get_quadrature_nodes)
    .method("get_dofs_coordinates", &Laplacian_3D_Order1::get_dofs_coordinates)
    // setters
    .method("set_dirichlet_bc",     &Laplacian_3D_Order1::set_dirichlet_bc)
    .method("set_forcing_term",     &Laplacian_3D_Order1::set_forcing_term);
}
