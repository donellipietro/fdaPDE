// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include <thread>
#include <fdaPDE/core/utils/Symbols.h>
#include <fdaPDE/models/functional/fSRPDE.h>
using fdaPDE::models::FSRPDE;
#include <fdaPDE/core/utils/DataStructures/BlockFrame.h>
#include <fdaPDE/models/ModelTraits.h> 
#include <fdaPDE/core/FEM/PDE.h>
using fdaPDE::core::FEM::DefaultOperator;
using fdaPDE::core::FEM::PDE;
#include <fdaPDE/core/MESH/Mesh.h>
using fdaPDE::core::MESH::Mesh;
#include <fdaPDE/models/SamplingDesign.h>
#include <fdaPDE/core/MESH/engines/AlternatingDigitalTree/ADT.h>
using fdaPDE::core::MESH::ADT;

#include "Common.h"

RCPP_EXPOSED_AS  (Laplacian_2D_Order1)
RCPP_EXPOSED_WRAP(Laplacian_2D_Order1)
RCPP_EXPOSED_AS  (Laplacian_3D_Order1)
RCPP_EXPOSED_WRAP(Laplacian_3D_Order1)

// expose RegularizingPDE as possible argument to other Rcpp modules
RCPP_EXPOSED_AS  (ConstantCoefficients_2D_Order1)
RCPP_EXPOSED_WRAP(ConstantCoefficients_2D_Order1)
// expose RegularizingPDE as possible argument to other Rcpp modules
RCPP_EXPOSED_AS  (SpaceVarying_2D_Order1)
RCPP_EXPOSED_WRAP(SpaceVarying_2D_Order1)

// wrapper for FSRPDE module
template <typename RegularizingPDE, typename S> class R_FSRPDE {
protected:
  typedef RegularizingPDE RegularizingPDE_;
  RegularizingPDE_ regularization_; // tipo di PDE

  FSRPDE<typename RegularizingPDE_::PDEType, fdaPDE::models::SpaceOnly, S> model_;
  BlockFrame<double, int> df_;
public:
  
  R_FSRPDE(const RegularizingPDE_ &regularization)
    : regularization_(regularization) {
    model_.setPDE(regularization_.pde());
  };

  // initialization
  void init() { model_.init(); }
  
  // setters
  void set_lambda_s(double lambdaS) { model_.setLambdaS(lambdaS); }
  // void set_lambdas(std::vector<double> lambdas) {
  //  std::vector<SVector<1>> l_;
  //  for(auto v : lambdas) l_.push_back(SVector<1>(v));
  //  model_.setLambda(l_);
  //}
  void set_observations(const DMatrix<double>& data) {
    df_.template insert<double>(OBSERVATIONS_BLK, data);
  }
  void set_locations(const DMatrix<double>& data) {
    model_.set_spatial_locations(data);
  }

  // getters
  DMatrix<double> f() const { return model_.f(); }
  DMatrix<double> fitted() const { return model_.fitted(); }
  // SpMatrix<double> R0() const { return model_.R0(); }
  // SpMatrix<double> Psi() { return model_.Psi(not_nan()); }
    
  // initialize model and solve smoothing problem
  void solve() {
    model_.setData(df_);
    model_.init();
    model_.solve();
    return;
  }

};

// definition of Rcpp module
typedef R_FSRPDE<Laplacian_2D_Order1, fdaPDE::models::GeoStatMeshNodes> FSRPDE_Laplacian_2D_GeoStatNodes;

// locations == nodes

RCPP_MODULE(FSRPDE_Laplacian_2D_GeoStatNodes) {
  Rcpp::class_<FSRPDE_Laplacian_2D_GeoStatNodes>("FSRPDE_Laplacian_2D_GeoStatNodes")
    .constructor<Laplacian_2D_Order1>()
    // getters
    .method("f",                        &FSRPDE_Laplacian_2D_GeoStatNodes::loadings)
    .method("fitted",                   &FSRPDE_Laplacian_2D_GeoStatNodes::scores)
    // setters
    .method("set_lambda_s",             &FSRPDE_Laplacian_2D_GeoStatNodes::set_lambda_s)
    // .method("set_lambdas",              &FSRPDE_Laplacian_2D_GeoStatNodes::set_lambdas)
    .method("set_observations",         &FSRPDE_Laplacian_2D_GeoStatNodes::set_observations)
    // .method("R0",                       &FSRPDE_Laplacian_2D_GeoStatNodes::R0)
    .method("init",                     &FSRPDE_Laplacian_2D_GeoStatNodes::init)
    // .method("init_regularization",      &FSRPDE_Laplacian_2D_GeoStatNodes::init_regularization)
    // .method("init_pde",                 &FSRPDE_Laplacian_2D_GeoStatNodes::init_pde)
    // .method("Psi",                      &FSRPDE_Laplacian_2D_GeoStatNodes::Psi)
    .method("solve",                    &FSRPDE_Laplacian_2D_GeoStatNodes::solve);
}

// locations != nodes

typedef R_FSRPDE<Laplacian_2D_Order1, fdaPDE::models::GeoStatLocations> FSRPDE_Laplacian_2D_GeoStatLocations;
RCPP_MODULE(FSRPDE_Laplacian_2D_GeoStatLocations) {
  Rcpp::class_<FSRPDE_Laplacian_2D_GeoStatLocations>("FSRPDE_Laplacian_2D_GeoStatLocations")
    .constructor<Laplacian_2D_Order1>()
    // getters
    .method("loadings",                 &FSRPDE_Laplacian_2D_GeoStatLocations::loadings)
    .method("scores",                   &FSRPDE_Laplacian_2D_GeoStatLocations::scores)
    // setters
    .method("set_lambda_s",             &FSRPDE_Laplacian_2D_GeoStatLocations::set_lambda_s)
    // .method("set_lambdas",              &FSRPDE_Laplacian_2D_GeoStatLocations::set_lambdas)
    .method("set_locations",            &FSRPDE_Laplacian_2D_GeoStatLocations::set_locations)
    .method("set_observations",         &FSRPDE_Laplacian_2D_GeoStatLocations::set_observations)
    // .method("R0",                       &FSRPDE_Laplacian_2D_GeoStatLocations::R0)
    // .method("init_regularization",      &FSRPDE_Laplacian_2D_GeoStatLocations::init_regularization)
    // .method("init_pde",                 &FSRPDE_Laplacian_2D_GeoStatLocations::init_pde)
    .method("init",                     &FSRPDE_Laplacian_2D_GeoStatLocations::init)
    // .method("Psi",                      &FSRPDE_Laplacian_2D_GeoStatLocations::Psi)
    .method("solve",                    &FSRPDE_Laplacian_2D_GeoStatLocations::solve);
}