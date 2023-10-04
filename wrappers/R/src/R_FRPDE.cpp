// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include <thread>
#include <fdaPDE/core/utils/Symbols.h>
#include <fdaPDE/models/functional/FRPDE.h>
using fdaPDE::models::FRPDE;
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
RCPP_EXPOSED_AS(Laplacian_Surface_Order1)
RCPP_EXPOSED_WRAP(Laplacian_Surface_Order1)
RCPP_EXPOSED_AS(Laplacian_3D_Order1)
RCPP_EXPOSED_WRAP(Laplacian_3D_Order1)

// expose RegularizingPDE as possible argument to other Rcpp modules
RCPP_EXPOSED_AS(ConstantCoefficients_2D_Order1)
RCPP_EXPOSED_WRAP(ConstantCoefficients_2D_Order1)
// expose RegularizingPDE as possible argument to other Rcpp modules
RCPP_EXPOSED_AS(SpaceVarying_2D_Order1)
RCPP_EXPOSED_WRAP(SpaceVarying_2D_Order1)

// wrapper for FRPDE module
template <typename PDE, typename SamplingDesign>
class R_FRPDE
{
protected:
  typedef PDE RegularizingPDE_;
  RegularizingPDE_ regularization_;

  FRPDE<typename RegularizingPDE_::PDEType, SamplingDesign> model_;

public:
  // Constructor
  R_FRPDE(const RegularizingPDE_ &regularization)
      : regularization_(regularization)
  {
    model_.setPDE(regularization_.pde());
  };

  // Initialization
  void init() { model_.init(); }

  // Setters
  void set_lambda_s(double lambdaS) { model_.setLambdaS(lambdaS); }
  // void set_lambdas(std::vector<double> lambdas) {
  //  std::vector<SVector<1>> l_;
  //  for(auto v : lambdas) l_.push_back(SVector<1>(v));
  //  model_.setLambda(l_);
  //}
  void set_observations(const DMatrix<double> &data)
  {
    // df_.template insert<double>(OBSERVATIONS_BLK, data);
    model_.setData(data);
  }
  void set_locations(const DMatrix<double> &data)
  {
    model_.set_spatial_locations(data);
  }
  void set_verbose(bool verbose)
  {
    model_.set_verbose(verbose);
  }

  // getters
  DMatrix<double> f() const { return model_.f(); }
  DMatrix<double> fitted() const { return model_.fitted(); }
  // SpMatrix<double> R0() const { return model_.R0(); }
  // SpMatrix<double> Psi() { return model_.Psi(not_nan()); }

  // Solve method
  void solve()
  {
    model_.init();
    model_.solve();
    return;
  }

  // Tune method
  SVector<1> tune(std::vector<double> &lambdas)
  {
    std::vector<SVector<1>> l_;
    for (auto v : lambdas)
      l_.push_back(SVector<1>(v));

    model_.init();
    return model_.tune(l_);
  }
};

// Laplacian_2D_Order1, locations == nodes
typedef R_FRPDE<Laplacian_2D_Order1, fdaPDE::models::GeoStatMeshNodes> FRPDE_Laplacian_2D_GeoStatNodes;
RCPP_MODULE(FRPDE_Laplacian_2D_GeoStatNodes)
{
  Rcpp::class_<FRPDE_Laplacian_2D_GeoStatNodes>("FRPDE_Laplacian_2D_GeoStatNodes")
      .constructor<Laplacian_2D_Order1>()
      // Initializations
      .method("init", &FRPDE_Laplacian_2D_GeoStatNodes::init)
      // .method("init_regularization", &FRPDE_Laplacian_2D_GeoStatNodes::init_regularization)
      //.method("init_pde", &FRPDE_Laplacian_2D_GeoStatNodes::init_pde)
      // Getters
      .method("f", &FRPDE_Laplacian_2D_GeoStatNodes::f)
      .method("fitted", &FRPDE_Laplacian_2D_GeoStatNodes::fitted)
      // .method("R0", &FRPDE_Laplacian_2D_GeoStatNodes::R0)
      // .method("Psi", &FRPDE_Laplacian_2D_GeoStatNodes::Psi)
      // Setters
      .method("set_lambda_s", &FRPDE_Laplacian_2D_GeoStatNodes::set_lambda_s)
      // .method("set_lambdas", &FRPDE_Laplacian_2D_GeoStatNodes::set_lambdas)
      .method("set_observations", &FRPDE_Laplacian_2D_GeoStatNodes::set_observations)
      .method("set_verbose", &FRPDE_Laplacian_2D_GeoStatNodes::set_verbose)
      // Solve method
      .method("solve", &FRPDE_Laplacian_2D_GeoStatNodes::solve)
      // Tune method
      .method("tune", &FRPDE_Laplacian_2D_GeoStatNodes::tune);
}

// Laplacian_2D_Order1, locations != nodes
typedef R_FRPDE<Laplacian_2D_Order1, fdaPDE::models::GeoStatLocations> FRPDE_Laplacian_2D_GeoStatLocations;
RCPP_MODULE(FRPDE_Laplacian_2D_GeoStatLocations)
{
  Rcpp::class_<FRPDE_Laplacian_2D_GeoStatLocations>("FRPDE_Laplacian_2D_GeoStatLocations")
      .constructor<Laplacian_2D_Order1>()
      // Initializations
      .method("init", &FRPDE_Laplacian_2D_GeoStatLocations::init)
      // .method("init_regularization", &FRPDE_Laplacian_2D_GeoStatLocations::init_regularization)
      // .method("init_pde", &FRPDE_Laplacian_2D_GeoStatLocations::init_pde)
      // Getters
      .method("f", &FRPDE_Laplacian_2D_GeoStatLocations::f)
      .method("fitted", &FRPDE_Laplacian_2D_GeoStatLocations::fitted)
      // .method("R0", &FRPDE_Laplacian_2D_GeoStatLocations::R0)
      // .method("Psi", &FRPDE_Laplacian_2D_GeoStatLocations::Psi)
      // Setters
      .method("set_lambda_s", &FRPDE_Laplacian_2D_GeoStatLocations::set_lambda_s)
      // .method("set_lambdas", &FRPDE_Laplacian_2D_GeoStatLocations::set_lambdas)
      .method("set_locations", &FRPDE_Laplacian_2D_GeoStatLocations::set_locations)
      .method("set_observations", &FRPDE_Laplacian_2D_GeoStatLocations::set_observations)
      .method("set_verbose", &FRPDE_Laplacian_2D_GeoStatLocations::set_verbose)
      // Solve method
      .method("solve", &FRPDE_Laplacian_2D_GeoStatLocations::solve)
      // Tune method
      .method("tune", &FRPDE_Laplacian_2D_GeoStatLocations::tune);
}

// Laplacian_Surface_Order1, locations == nodes
typedef R_FRPDE<Laplacian_Surface_Order1, fdaPDE::models::GeoStatMeshNodes> FRPDE_Laplacian_Surface_GeoStatNodes;
RCPP_MODULE(FRPDE_Laplacian_Surface_GeoStatNodes)
{
  Rcpp::class_<FRPDE_Laplacian_Surface_GeoStatNodes>("FRPDE_Laplacian_Surface_GeoStatNodes")
      .constructor<Laplacian_Surface_Order1>()
      // Initializations
      .method("init", &FRPDE_Laplacian_Surface_GeoStatNodes::init)
      // .method("init_regularization", &FRPDE_Laplacian_Surface_GeoStatNodes::init_regularization)
      //.method("init_pde", &FRPDE_Laplacian_Surface_GeoStatNodes::init_pde)
      // Getters
      .method("f", &FRPDE_Laplacian_Surface_GeoStatNodes::f)
      .method("fitted", &FRPDE_Laplacian_Surface_GeoStatNodes::fitted)
      // .method("R0", &FRPDE_Laplacian_Surface_GeoStatNodes::R0)
      // .method("Psi", &FRPDE_Laplacian_Surface_GeoStatNodes::Psi)
      // Setters
      .method("set_lambda_s", &FRPDE_Laplacian_Surface_GeoStatNodes::set_lambda_s)
      // .method("set_lambdas", &FRPDE_Laplacian_Surface_GeoStatNodes::set_lambdas)
      .method("set_observations", &FRPDE_Laplacian_Surface_GeoStatNodes::set_observations)
      .method("set_verbose", &FRPDE_Laplacian_Surface_GeoStatNodes::set_verbose)
      // Solve method
      .method("solve", &FRPDE_Laplacian_Surface_GeoStatNodes::solve)
      // Tune method
      .method("tune", &FRPDE_Laplacian_Surface_GeoStatNodes::tune);
}

// Laplacian_Surface_Order1, locations != nodes
typedef R_FRPDE<Laplacian_Surface_Order1, fdaPDE::models::GeoStatLocations> FRPDE_Laplacian_Surface_GeoStatLocations;
RCPP_MODULE(FRPDE_Laplacian_Surface_GeoStatLocations)
{
  Rcpp::class_<FRPDE_Laplacian_Surface_GeoStatLocations>("FRPDE_Laplacian_Surface_GeoStatLocations")
      .constructor<Laplacian_Surface_Order1>()
      // Initializations
      .method("init", &FRPDE_Laplacian_Surface_GeoStatLocations::init)
      // .method("init_regularization", &FRPDE_Laplacian_Surface_GeoStatLocations::init_regularization)
      // .method("init_pde", &FRPDE_Laplacian_Surface_GeoStatLocations::init_pde)
      // Getters
      .method("f", &FRPDE_Laplacian_Surface_GeoStatLocations::f)
      .method("fitted", &FRPDE_Laplacian_Surface_GeoStatLocations::fitted)
      // .method("R0", &FRPDE_Laplacian_Surface_GeoStatLocations::R0)
      // .method("Psi", &FRPDE_Laplacian_Surface_GeoStatLocations::Psi)
      // Setters
      .method("set_lambda_s", &FRPDE_Laplacian_Surface_GeoStatLocations::set_lambda_s)
      // .method("set_lambdas", &FRPDE_Laplacian_Surface_GeoStatLocations::set_lambdas)
      .method("set_locations", &FRPDE_Laplacian_Surface_GeoStatLocations::set_locations)
      .method("set_observations", &FRPDE_Laplacian_Surface_GeoStatLocations::set_observations)
      .method("set_verbose", &FRPDE_Laplacian_Surface_GeoStatLocations::set_verbose)
      // Solve method
      .method("solve", &FRPDE_Laplacian_Surface_GeoStatLocations::solve)
      // Tune method
      .method("tune", &FRPDE_Laplacian_Surface_GeoStatLocations::tune);
}