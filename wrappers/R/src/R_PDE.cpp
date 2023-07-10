#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include <fdaPDE/core/utils/Symbols.h>
#include <fdaPDE/core/MESH/Mesh.h>
using fdaPDE::core::MESH::Mesh;
#include <fdaPDE/core/FEM/PDE.h>
using fdaPDE::core::FEM::PDE;
using fdaPDE::core::FEM::PDEBase;
using fdaPDE::core::FEM::DefaultOperator;
#include <fdaPDE/core/FEM/operators/SpaceVaryingFunctors.h>
using fdaPDE::core::FEM::SpaceVaryingAdvection;
using fdaPDE::core::FEM::SpaceVaryingDiffusion;
using fdaPDE::core::FEM::SpaceVaryingReaction;

// R wrapper for Partial Differential Equations Lf = u
class R_PDE {
private:
  Mesh<2,2,1> domain_;
  PDEBase* pde_; // pointer to PDE object
public:
  // constructor
  R_PDE(const Rcpp::List& mesh_data, const Rcpp::List& pde_data) :
    // initialize domain
    domain_(Rcpp::as<DMatrix<double>>(mesh_data["nodes"]),
	    Rcpp::as<DMatrix<int>>   (mesh_data["edges"]),
	    Rcpp::as<DMatrix<int>>   (mesh_data["elements"]),
	    Rcpp::as<DMatrix<int>>   (mesh_data["neigh"]),
	    Rcpp::as<DMatrix<int>>   (mesh_data["boundary"])){
    // initialize pde depending on run-time informations
    auto L = Laplacian();
    DMatrix<double> u = DMatrix<double>::Zero(domain_.elements()*3, 1);
    pde_ = new PDE<2,2,1, decltype(L), DMatrix<double>>(domain_, L, u);
  }

  
  
  // destructor
  ~R_PDE() { delete pde_; }
  
};
  
