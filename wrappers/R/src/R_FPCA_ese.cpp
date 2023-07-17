// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include <thread>
#include <fdaPDE/core/utils/Symbols.h>
#include <fdaPDE/models/functional/fPCA.h>
using fdaPDE::models::FPCA;
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

// wrapper for SRPDE module
template <typename RegularizingPDE, typename S> class R_FPCA {
protected:
  typedef RegularizingPDE RegularizingPDE_;
  RegularizingPDE_ regularization_; // tipo di PDE

  // qui metti il modello che vuoi wrappare (FPLS??)
  FPCA<typename RegularizingPDE_::PDEType, fdaPDE::models::SpaceOnly, S, fdaPDE::models::gcv_lambda_selection> model_;
  BlockFrame<double, int> df_;
public:
  
  R_FPCA(const RegularizingPDE_ &regularization)
    : regularization_(regularization) {
    model_.setPDE(regularization_.pde());
  };

  // metodi di init, messi a caso giusto nel caso dovessero servire

  
  void init_pde() { model_.init_pde(); }
  void init() { model_.init(); }
  void init_regularization() { model_.init_pde(); model_.init_regularization(); model_.init_sampling(); }
  
  /* setters */
  void set_lambda_s(double lambdaS) { model_.setLambdaS(lambdaS); }
  void set_lambdas(std::vector<double> lambdas) {
    std::vector<SVector<1>> l_;
    for(auto v : lambdas) l_.push_back(SVector<1>(v));
    model_.setLambda(l_);
  }

  void set_npc(std::size_t n) { model_.set_npc(n); } // numero di componenti principali
  
  void set_observations(const DMatrix<double>& data) {
    df_.template insert<double>(OBSERVATIONS_BLK, data);
  }
  void set_locations(const DMatrix<double>& data) {
    model_.set_spatial_locations(data);
  }
  /* getters */
  DMatrix<double> loadings() const { return model_.loadings(); }
  DMatrix<double> scores() const { return model_.scores(); }

  SpMatrix<double> R0() const { return model_.R0(); }
  SpMatrix<double> Psi() { return model_.Psi(not_nan()); }
    
  /* initialize model and solve smoothing problem */
  void solve() {
    model_.setData(df_);
    model_.init();
    model_.solve();
    return;
  }

  // metti quello che ti serve
};

// definition of Rcpp module
typedef R_FPCA<Laplacian_2D_Order1, fdaPDE::models::GeoStatMeshNodes> FPCA_Laplacian_2D_GeoStatNodes;

// locations == nodes

RCPP_MODULE(FPCA_Laplacian_2D_GeoStatNodes) {
  Rcpp::class_<FPCA_Laplacian_2D_GeoStatNodes>("FPCA_Laplacian_2D_GeoStatNodes")
    .constructor<Laplacian_2D_Order1>()
    // getters
    .method("loadings",         &FPCA_Laplacian_2D_GeoStatNodes::loadings)
    .method("scores",           &FPCA_Laplacian_2D_GeoStatNodes::scores)
    // setters
    .method("set_lambda_s",     &FPCA_Laplacian_2D_GeoStatNodes::set_lambda_s)
    .method("set_lambdas",     &FPCA_Laplacian_2D_GeoStatNodes::set_lambdas)
    .method("set_npc", &FPCA_Laplacian_2D_GeoStatNodes::set_npc)
    .method("set_observations", &FPCA_Laplacian_2D_GeoStatNodes::set_observations)
    .method("R0",       &FPCA_Laplacian_2D_GeoStatNodes::R0)
    .method("init",       &FPCA_Laplacian_2D_GeoStatNodes::init)
    .method("init_regularization",       &FPCA_Laplacian_2D_GeoStatNodes::init_regularization)
    .method("init_pde",       &FPCA_Laplacian_2D_GeoStatNodes::init_pde)
    .method("Psi",       &FPCA_Laplacian_2D_GeoStatNodes::Psi)
    .method("solve",            &FPCA_Laplacian_2D_GeoStatNodes::solve);
}

// locations != nodes

typedef R_FPCA<Laplacian_2D_Order1, fdaPDE::models::GeoStatLocations> FPCA_Laplacian_2D_GeoStatLocations;
RCPP_MODULE(FPCA_Laplacian_2D_GeoStatLocations) {
  Rcpp::class_<FPCA_Laplacian_2D_GeoStatLocations>("FPCA_Laplacian_2D_GeoStatLocations")
    .constructor<Laplacian_2D_Order1>()
    // getters
    .method("loadings",         &FPCA_Laplacian_2D_GeoStatLocations::loadings)
    .method("scores",           &FPCA_Laplacian_2D_GeoStatLocations::scores)
    // setters
    .method("set_lambda_s",     &FPCA_Laplacian_2D_GeoStatLocations::set_lambda_s)
    .method("set_lambdas",     &FPCA_Laplacian_2D_GeoStatLocations::set_lambdas)
    .method("set_npc", &FPCA_Laplacian_2D_GeoStatLocations::set_npc)
    .method("set_locations",  &FPCA_Laplacian_2D_GeoStatLocations::set_locations)
    .method("set_observations", &FPCA_Laplacian_2D_GeoStatLocations::set_observations)
    .method("R0",       &FPCA_Laplacian_2D_GeoStatLocations::R0)
    .method("init_regularization",       &FPCA_Laplacian_2D_GeoStatLocations::init_regularization)
    .method("init_pde",       &FPCA_Laplacian_2D_GeoStatLocations::init_pde)
    .method("init",       &FPCA_Laplacian_2D_GeoStatLocations::init)
    .method("Psi",       &FPCA_Laplacian_2D_GeoStatLocations::Psi)
    .method("solve",            &FPCA_Laplacian_2D_GeoStatLocations::solve);
}

// 3D wrapper
typedef R_FPCA<Laplacian_3D_Order1, fdaPDE::models::GeoStatMeshNodes> FPCA_Laplacian_3D_GeoStatNodes;
RCPP_MODULE(FPCA_Laplacian_3D_GeoStatNodes) {
  Rcpp::class_<FPCA_Laplacian_3D_GeoStatNodes>("FPCA_Laplacian_3D_GeoStatNodes")
    .constructor<Laplacian_3D_Order1>()
    // getters
    .method("loadings",         &FPCA_Laplacian_3D_GeoStatNodes::loadings)
    .method("scores",           &FPCA_Laplacian_3D_GeoStatNodes::scores)
    // setters
    .method("set_lambda_s",     &FPCA_Laplacian_3D_GeoStatNodes::set_lambda_s)
    .method("set_observations", &FPCA_Laplacian_3D_GeoStatNodes::set_observations)
    .method("init_regularization",       &FPCA_Laplacian_3D_GeoStatNodes::init_regularization)
    .method("init_pde",       &FPCA_Laplacian_3D_GeoStatNodes::init_pde)
    .method("R0",       &FPCA_Laplacian_3D_GeoStatNodes::R0)
    // solve method
    .method("solve",            &FPCA_Laplacian_3D_GeoStatNodes::solve);
}
