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

RCPP_EXPOSED_AS(Laplacian_2D_Order1)
RCPP_EXPOSED_WRAP(Laplacian_2D_Order1)
RCPP_EXPOSED_AS(Laplacian_3D_Order1)
RCPP_EXPOSED_WRAP(Laplacian_3D_Order1)

// expose RegularizingPDE as possible argument to other Rcpp modules
RCPP_EXPOSED_AS(ConstantCoefficients_2D_Order1)
RCPP_EXPOSED_WRAP(ConstantCoefficients_2D_Order1)
// expose RegularizingPDE as possible argument to other Rcpp modules
RCPP_EXPOSED_AS(SpaceVarying_2D_Order1)
RCPP_EXPOSED_WRAP(SpaceVarying_2D_Order1)

// wrapper for fPCA module
template <typename RegularizingPDE, typename S>
class R_FPCA
{
protected:
  typedef RegularizingPDE RegularizingPDE_;
  RegularizingPDE_ regularization_;

  FPCA<typename RegularizingPDE_::PDEType, fdaPDE::models::SpaceOnly, S, fdaPDE::models::gcv_lambda_selection> model_;
  BlockFrame<double, int> df_;

public:
  // Constructor
  R_FPCA(const RegularizingPDE_ &regularization)
      : regularization_(regularization)
  {
    model_.setPDE(regularization_.pde());
  };

  // Initializations
  void init_pde() { model_.init_pde(); }
  void init() { model_.init(); }
  void init_regularization()
  {
    model_.init_pde();
    model_.init_regularization();
    model_.init_sampling();
  }

  // Getters
  DMatrix<double> W() const { return model_.W(); }
  DMatrix<double> loadings() const { return model_.loadings(); }
  DMatrix<double> scores() const { return model_.scores(); }
  SpMatrix<double> R0() const { return model_.R0(); }
  SpMatrix<double> Psi() { return model_.Psi(not_nan()); }
  std::vector<double> lambda_opt()
  {
    std::vector<double> lambda;
    for (const auto v : model_.lambda_opt())
      lambda.push_back(v[0]);
    return lambda;
  }

  // Setters
  void set_npc(std::size_t n) { model_.set_npc(n); }
  void set_lambdas(std::vector<double> lambdas)
  {
    std::vector<SVector<1>> l_;
    for (auto v : lambdas)
      l_.push_back(SVector<1>(v));
    model_.setLambda(l_);
  }
  void set_observations(const DMatrix<double> &data)
  {
    df_.template insert<double>(OBSERVATIONS_BLK, data);
  }
  void set_locations(const DMatrix<double> &data)
  {
    model_.set_spatial_locations(data);
  }

  // Solve method
  void solve()
  {
    model_.setData(df_);
    model_.init();
    model_.solve();
    return;
  }
};

// Locations == Nodes
typedef R_FPCA<Laplacian_2D_Order1, fdaPDE::models::GeoStatMeshNodes> FPCA_Laplacian_2D_GeoStatNodes_GCV;
RCPP_MODULE(FPCA_Laplacian_2D_GeoStatNodes_GCV)
{
  Rcpp::class_<FPCA_Laplacian_2D_GeoStatNodes_GCV>("FPCA_Laplacian_2D_GeoStatNodes_GCV")
      .constructor<Laplacian_2D_Order1>()
      // Initializations
      .method("init", &FPCA_Laplacian_2D_GeoStatNodes_GCV::init)
      .method("init_regularization", &FPCA_Laplacian_2D_GeoStatNodes_GCV::init_regularization)
      .method("init_pde", &FPCA_Laplacian_2D_GeoStatNodes_GCV::init_pde)
      // Getters
      .method("W", &FPCA_Laplacian_2D_GeoStatNodes_GCV::W)
      .method("loadings", &FPCA_Laplacian_2D_GeoStatNodes_GCV::loadings)
      .method("scores", &FPCA_Laplacian_2D_GeoStatNodes_GCV::scores)
      .method("R0", &FPCA_Laplacian_2D_GeoStatNodes_GCV::R0)
      .method("Psi", &FPCA_Laplacian_2D_GeoStatNodes_GCV::Psi)
      .method("lambda_opt", &FPCA_Laplacian_2D_GeoStatNodes_GCV::lambda_opt)
      // Setters
      .method("set_lambdas", &FPCA_Laplacian_2D_GeoStatNodes_GCV::set_lambdas)
      .method("set_npc", &FPCA_Laplacian_2D_GeoStatNodes_GCV::set_npc)
      .method("set_observations", &FPCA_Laplacian_2D_GeoStatNodes_GCV::set_observations)
      // Solve method
      .method("solve", &FPCA_Laplacian_2D_GeoStatNodes_GCV::solve);
}

// Locations != Nodes
typedef R_FPCA<Laplacian_2D_Order1, fdaPDE::models::GeoStatLocations> FPCA_Laplacian_2D_GeoStatLocations_GCV;
RCPP_MODULE(FPCA_Laplacian_2D_GeoStatLocations_GCV)
{
  Rcpp::class_<FPCA_Laplacian_2D_GeoStatLocations_GCV>("FPCA_Laplacian_2D_GeoStatLocations_GCV")
      .constructor<Laplacian_2D_Order1>()
      // Initializations
      .method("init_regularization", &FPCA_Laplacian_2D_GeoStatLocations_GCV::init_regularization)
      .method("init_pde", &FPCA_Laplacian_2D_GeoStatLocations_GCV::init_pde)
      .method("init", &FPCA_Laplacian_2D_GeoStatLocations_GCV::init)
      // Getters
      .method("W", &FPCA_Laplacian_2D_GeoStatLocations_GCV::W)
      .method("loadings", &FPCA_Laplacian_2D_GeoStatLocations_GCV::loadings)
      .method("scores", &FPCA_Laplacian_2D_GeoStatLocations_GCV::scores)
      .method("R0", &FPCA_Laplacian_2D_GeoStatLocations_GCV::R0)
      .method("Psi", &FPCA_Laplacian_2D_GeoStatLocations_GCV::Psi)
      .method("lambda_opt", &FPCA_Laplacian_2D_GeoStatLocations_GCV::lambda_opt)
      // Setters
      .method("set_lambdas", &FPCA_Laplacian_2D_GeoStatLocations_GCV::set_lambdas)
      .method("set_npc", &FPCA_Laplacian_2D_GeoStatLocations_GCV::set_npc)
      .method("set_locations", &FPCA_Laplacian_2D_GeoStatLocations_GCV::set_locations)
      .method("set_observations", &FPCA_Laplacian_2D_GeoStatLocations_GCV::set_observations)
      // Solve method
      .method("solve", &FPCA_Laplacian_2D_GeoStatLocations_GCV::solve);
}

// 3D wrapper
typedef R_FPCA<Laplacian_3D_Order1, fdaPDE::models::GeoStatMeshNodes> FPCA_Laplacian_3D_GeoStatNodes_GCV;
RCPP_MODULE(FPCA_Laplacian_3D_GeoStatNodes_GCV)
{
  Rcpp::class_<FPCA_Laplacian_3D_GeoStatNodes_GCV>("FPCA_Laplacian_3D_GeoStatNodes_GCV")
      .constructor<Laplacian_3D_Order1>()
      // Initializations
      .method("init", &FPCA_Laplacian_3D_GeoStatNodes_GCV::init)
      .method("init_regularization", &FPCA_Laplacian_3D_GeoStatNodes_GCV::init_regularization)
      .method("init_pde", &FPCA_Laplacian_3D_GeoStatNodes_GCV::init_pde)
      // Getters
      .method("W", &FPCA_Laplacian_3D_GeoStatNodes_GCV::W)
      .method("loadings", &FPCA_Laplacian_3D_GeoStatNodes_GCV::loadings)
      .method("scores", &FPCA_Laplacian_3D_GeoStatNodes_GCV::scores)
      .method("R0", &FPCA_Laplacian_3D_GeoStatNodes_GCV::R0)
      .method("Psi", &FPCA_Laplacian_3D_GeoStatNodes_GCV::Psi)
      .method("lambda_opt", &FPCA_Laplacian_3D_GeoStatNodes_GCV::lambda_opt)
      // Setters
      .method("set_lambdas", &FPCA_Laplacian_3D_GeoStatNodes_GCV::set_lambdas)
      .method("set_npc", &FPCA_Laplacian_3D_GeoStatNodes_GCV::set_npc)
      .method("set_observations", &FPCA_Laplacian_3D_GeoStatNodes_GCV::set_observations)
      // Solve method
      .method("solve", &FPCA_Laplacian_3D_GeoStatNodes_GCV::solve);
}
