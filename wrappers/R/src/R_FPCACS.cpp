// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include <thread>
#include <fdaPDE/core/utils/Symbols.h>
#include <fdaPDE/models/functional/fPCA_CS.h>
using fdaPDE::models::FPCA_CS;
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

// wrapper for fPCA Closed Solution module
template <typename RegularizingPDE, typename S>
class R_FPCA_CS
{
protected:
  typedef RegularizingPDE RegularizingPDE_;
  RegularizingPDE_ regularization_;

  FPCA_CS<typename RegularizingPDE_::PDEType, fdaPDE::models::SpaceOnly, S, fdaPDE::models::fixed_lambda> model_;
  BlockFrame<double, int> df_;

public:
  // Constructor
  R_FPCA_CS(const RegularizingPDE_ &regularization)
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
  DMatrix<double> coefficients() const { return model_.coefficients(); }
  SpMatrix<double> R0() const { return model_.R0(); }
  SpMatrix<double> Psi() { return model_.Psi(fdaPDE::models::not_nan()); }

  // Setters
  void set_npc(std::size_t n) { model_.set_npc(n); }
  void set_verbose(bool verbose) { model_.set_verbose(verbose); }
  void set_mass_lumping(bool mass_lumping) { model_.set_mass_lumping(mass_lumping); }
  void set_iterative(bool iterative) { model_.set_iterative(iterative); }
  void set_coefficients_position(unsigned int coefficients_position) { model_.set_coefficients_position(coefficients_position); }
  void set_lambda_s(double lambdaS) { model_.setLambdaS(lambdaS); }
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

// fPCA Closed Solution Rcpp module

// Locations == Nodes
typedef R_FPCA_CS<Laplacian_2D_Order1, fdaPDE::models::GeoStatMeshNodes> FPCA_CS_Laplacian_2D_GeoStatNodes;
RCPP_MODULE(FPCA_CS_Laplacian_2D_GeoStatNodes)
{
  Rcpp::class_<FPCA_CS_Laplacian_2D_GeoStatNodes>("FPCA_CS_Laplacian_2D_GeoStatNodes")
      .constructor<Laplacian_2D_Order1>()
      // Initializations
      .method("init", &FPCA_CS_Laplacian_2D_GeoStatNodes::init)
      .method("init_regularization", &FPCA_CS_Laplacian_2D_GeoStatNodes::init_regularization)
      .method("init_pde", &FPCA_CS_Laplacian_2D_GeoStatNodes::init_pde)
      // Getters
      .method("W", &FPCA_CS_Laplacian_2D_GeoStatNodes::W)
      .method("loadings", &FPCA_CS_Laplacian_2D_GeoStatNodes::loadings)
      .method("scores", &FPCA_CS_Laplacian_2D_GeoStatNodes::scores)
      .method("coefficients", &FPCA_CS_Laplacian_2D_GeoStatNodes::coefficients)
      .method("R0", &FPCA_CS_Laplacian_2D_GeoStatNodes::R0)
      .method("Psi", &FPCA_CS_Laplacian_2D_GeoStatNodes::Psi)
      // Setters
      .method("set_lambda_s", &FPCA_CS_Laplacian_2D_GeoStatNodes::set_lambda_s)
      .method("set_lambdas", &FPCA_CS_Laplacian_2D_GeoStatNodes::set_lambdas)
      .method("set_npc", &FPCA_CS_Laplacian_2D_GeoStatNodes::set_npc)
      .method("set_observations", &FPCA_CS_Laplacian_2D_GeoStatNodes::set_observations)
      .method("set_verbose", &FPCA_CS_Laplacian_2D_GeoStatNodes::set_verbose)
      .method("set_mass_lumping", &FPCA_CS_Laplacian_2D_GeoStatNodes::set_mass_lumping)
      .method("set_iterative", &FPCA_CS_Laplacian_2D_GeoStatNodes::set_iterative)
      .method("set_coefficients_position", &FPCA_CS_Laplacian_2D_GeoStatNodes::set_coefficients_position)
      // Solve method
      .method("solve", &FPCA_CS_Laplacian_2D_GeoStatNodes::solve);
}

// Locations != Nodes
typedef R_FPCA_CS<Laplacian_2D_Order1, fdaPDE::models::GeoStatLocations> FPCA_CS_Laplacian_2D_GeoStatLocations;
RCPP_MODULE(FPCA_CS_Laplacian_2D_GeoStatLocations)
{
  Rcpp::class_<FPCA_CS_Laplacian_2D_GeoStatLocations>("FPCA_CS_Laplacian_2D_GeoStatLocations")
      .constructor<Laplacian_2D_Order1>()
      // Initializations
      .method("init_regularization", &FPCA_CS_Laplacian_2D_GeoStatLocations::init_regularization)
      .method("init_pde", &FPCA_CS_Laplacian_2D_GeoStatLocations::init_pde)
      .method("init", &FPCA_CS_Laplacian_2D_GeoStatLocations::init)
      // Getters
      .method("W", &FPCA_CS_Laplacian_2D_GeoStatLocations::W)
      .method("loadings", &FPCA_CS_Laplacian_2D_GeoStatLocations::loadings)
      .method("scores", &FPCA_CS_Laplacian_2D_GeoStatLocations::scores)
      .method("coefficients", &FPCA_CS_Laplacian_2D_GeoStatLocations::coefficients)
      .method("R0", &FPCA_CS_Laplacian_2D_GeoStatLocations::R0)
      .method("Psi", &FPCA_CS_Laplacian_2D_GeoStatLocations::Psi)
      // Setters
      .method("set_lambda_s", &FPCA_CS_Laplacian_2D_GeoStatLocations::set_lambda_s)
      .method("set_lambdas", &FPCA_CS_Laplacian_2D_GeoStatLocations::set_lambdas)
      .method("set_npc", &FPCA_CS_Laplacian_2D_GeoStatLocations::set_npc)
      .method("set_locations", &FPCA_CS_Laplacian_2D_GeoStatLocations::set_locations)
      .method("set_observations", &FPCA_CS_Laplacian_2D_GeoStatLocations::set_observations)
      .method("set_verbose", &FPCA_CS_Laplacian_2D_GeoStatLocations::set_verbose)
      .method("set_mass_lumping", &FPCA_CS_Laplacian_2D_GeoStatLocations::set_mass_lumping)
      .method("set_iterative", &FPCA_CS_Laplacian_2D_GeoStatLocations::set_iterative)
      .method("set_coefficients_position", &FPCA_CS_Laplacian_2D_GeoStatLocations::set_coefficients_position)
      // Solve method
      .method("solve", &FPCA_CS_Laplacian_2D_GeoStatLocations::solve);
}

// 3D wrapper
typedef R_FPCA_CS<Laplacian_3D_Order1, fdaPDE::models::GeoStatMeshNodes> FPCA_CS_Laplacian_3D_GeoStatNodes;
RCPP_MODULE(FPCA_CS_Laplacian_3D_GeoStatNodes)
{
  Rcpp::class_<FPCA_CS_Laplacian_3D_GeoStatNodes>("FPCA_CS_Laplacian_3D_GeoStatNodes")
      .constructor<Laplacian_3D_Order1>()
      // Initializations
      .method("init_regularization", &FPCA_CS_Laplacian_3D_GeoStatNodes::init_regularization)
      .method("init_pde", &FPCA_CS_Laplacian_3D_GeoStatNodes::init_pde)
      // Getters
      .method("W", &FPCA_CS_Laplacian_3D_GeoStatNodes::W)
      .method("loadings", &FPCA_CS_Laplacian_3D_GeoStatNodes::loadings)
      .method("scores", &FPCA_CS_Laplacian_3D_GeoStatNodes::scores)
      .method("coefficients", &FPCA_CS_Laplacian_3D_GeoStatNodes::coefficients)
      .method("R0", &FPCA_CS_Laplacian_3D_GeoStatNodes::R0)
      // Setters
      .method("set_lambda_s", &FPCA_CS_Laplacian_3D_GeoStatNodes::set_lambda_s)
      .method("set_observations", &FPCA_CS_Laplacian_3D_GeoStatNodes::set_observations)
      .method("set_verbose", &FPCA_CS_Laplacian_3D_GeoStatNodes::set_verbose)
      .method("set_mass_lumping", &FPCA_CS_Laplacian_3D_GeoStatNodes::set_mass_lumping)
      .method("set_iterative", &FPCA_CS_Laplacian_3D_GeoStatNodes::set_iterative)
      .method("set_coefficients_position", &FPCA_CS_Laplacian_3D_GeoStatNodes::set_coefficients_position)
      // Solve method
      .method("solve", &FPCA_CS_Laplacian_3D_GeoStatNodes::solve);
}
