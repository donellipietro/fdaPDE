#ifndef __COMMON_H__
#define __COMMON_H__

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

template <unsigned int M, unsigned int N, unsigned int R, typename F>
class RegularizingPDE {
private:
  typedef typename std::decay<F>::type BilinearFormType;
  // internal data
  Mesh<M,N,R> domain_;
  PDE<M,N,R, BilinearFormType, DMatrix<double>> pde_;
public:
  // constructor
  RegularizingPDE(const Rcpp::List& R_Mesh) :
    // initialize domain
    domain_(Rcpp::as<DMatrix<double>>(R_Mesh["nodes"]),
	    Rcpp::as<DMatrix<int>>   (R_Mesh["edges"]),
	    Rcpp::as<DMatrix<int>>   (R_Mesh["elements"]),
	    Rcpp::as<DMatrix<int>>   (R_Mesh["neigh"]),
	    Rcpp::as<DMatrix<int>>   (R_Mesh["boundary"])),
    pde_(domain_) { pde_.setBilinearForm(BilinearFormType()); };
  
  // setters
  void set_dirichlet_bc(const DMatrix<double>& data){ pde_.setDirichletBC(data); }
  void set_forcing_term(const DMatrix<double>& data){ pde_.setForcing(data); }
  // getters
  DMatrix<double> get_quadrature_nodes() const { return pde_.integrator().quadratureNodes(domain_); };
  DMatrix<double> get_dofs_coordinates() const { return domain_.dofCoords(); };
  SpMatrix<double> R0() const { return pde_.R0(); }
  void init() { pde_.init(); }
  
  const PDE<M,N,R, BilinearFormType, DMatrix<double>>& pde() const { return pde_; }
  PDE<M,N,R, BilinearFormType, DMatrix<double>>& pde() { return pde_; }
  
  // compile time informations
  typedef PDE<M,N,R, BilinearFormType, DMatrix<double>> PDEType;
};
// define 2D simple Laplacian regularization.
typedef RegularizingPDE<2,2,1, decltype( std::declval<Laplacian<DefaultOperator>>() )>
Laplacian_2D_Order1;
// 3D simple Laplacian regularization
typedef RegularizingPDE<3,3,1, decltype( std::declval<Laplacian<DefaultOperator>>() )>
Laplacian_3D_Order1;

// constant coefficients PDE type
template<unsigned int M>
using ConstantCoefficientsPDE =
  decltype( std::declval<Laplacian<SMatrix<M>>>() +
	    std::declval<Gradient <SVector<M>>>() +
	    std::declval<Identity <double>>() );

// specialization of RegularizingPDE for constant coefficients case
template <unsigned int M, unsigned int N, unsigned int R>
class RegularizingPDE<M,N,R, ConstantCoefficientsPDE<M>> {
private:
  typedef typename std::decay<ConstantCoefficientsPDE<M>>::type BilinearFormType;
  // internal data
  Mesh<M,N,R> domain_;
  PDE<M,N,R, BilinearFormType, DMatrix<double>> pde_;
public:
  // constructor
  RegularizingPDE(const Rcpp::List& R_Mesh) :
    // initialize domain
    domain_(Rcpp::as<DMatrix<double>>(R_Mesh["nodes"]),
	    Rcpp::as<DMatrix<int>>   (R_Mesh["edges"]),
	    Rcpp::as<DMatrix<int>>   (R_Mesh["elements"]),
	    Rcpp::as<DMatrix<int>>   (R_Mesh["neigh"]),
	    Rcpp::as<DMatrix<int>>   (R_Mesh["boundary"])),
    pde_(domain_) {};
  
  // setters
  void set_dirichlet_bc(const DMatrix<double>& data){ pde_.setDirichletBC(data); }
  void set_forcing_term(const DMatrix<double>& data){ pde_.setForcing(data); }
  void set_PDE_parameters(const Rcpp::List& data){
    SMatrix<M> K = Rcpp::as<DMatrix<double>>(data["diffusion"]);
    SVector<M> b = Rcpp::as<DVector<double>>(data["transport"]);
    double c = Rcpp::as<double>(data["reaction"]);
    BilinearFormType bilinearForm = Laplacian(K) + Gradient(b) + Identity(c);
    pde_.setBilinearForm(bilinearForm);  
  };

  // getters
  DMatrix<double> get_quadrature_nodes() const { return pde_.integrator().quadratureNodes(domain_); };
  DMatrix<double> get_dofs_coordinates() const { return domain_.dofCoords(); };
  const PDE<M,N,R, BilinearFormType, DMatrix<double>>& pde() { return pde_; }
  
  // compile time informations
  typedef PDE<M,N,R, BilinearFormType, DMatrix<double>> PDEType;
};

// define 2D costant coefficient PDE regularization.
typedef RegularizingPDE<2,2,1, ConstantCoefficientsPDE<2>> ConstantCoefficients_2D_Order1;

// space varying PDE type
template<unsigned int M>
using SpaceVaryingPDE =
  decltype( std::declval< Laplacian< decltype(std::declval<SpaceVaryingDiffusion<M>>().asParameter()) >>() +
	    std::declval< Gradient < decltype(std::declval<SpaceVaryingAdvection<M>>().asParameter()) >>() +
	    std::declval< Identity < decltype(std::declval<SpaceVaryingReaction>().asParameter()) >>() );

// specialization of RegularizingPDE for space varying
template <unsigned int M, unsigned int N, unsigned int R>
class RegularizingPDE<M,N,R, SpaceVaryingPDE<M>> {
private:
  typedef typename std::decay<SpaceVaryingPDE<M>>::type BilinearFormType;
  // internal data
  Mesh<M,N,R> domain_;
  PDE<M,N,R, BilinearFormType, DMatrix<double>> pde_;
  // space-varying functors
  SpaceVaryingDiffusion<M> diffusion_;
  SpaceVaryingAdvection<M> advection_;
  SpaceVaryingReaction     reaction_;
public:
  // constructor
  RegularizingPDE(const Rcpp::List& R_Mesh) :
    // initialize domain
    domain_(Rcpp::as<DMatrix<double>>(R_Mesh["nodes"]),
	    Rcpp::as<DMatrix<int>>   (R_Mesh["edges"]),
	    Rcpp::as<DMatrix<int>>   (R_Mesh["elements"]),
	    Rcpp::as<DMatrix<int>>   (R_Mesh["neigh"]),
	    Rcpp::as<DMatrix<int>>   (R_Mesh["boundary"])),
    pde_(domain_) {};
  
  // setters
  void set_dirichlet_bc(const DMatrix<double>& data){ pde_.setDirichletBC(data); }
  void set_forcing_term(const DMatrix<double>& data){ pde_.setForcing(data); }
  void set_PDE_parameters(const Rcpp::List& data){
    DMatrix<double> K = Rcpp::as<DMatrix<double>>(data["diffusion"]);
    diffusion_.setData(K);
    DMatrix<double> b = Rcpp::as<DMatrix<double>>(data["transport"]);
    advection_.setData(b);
    DMatrix<double> c = Rcpp::as<DMatrix<double>>(data["reaction"]);
    reaction_.setData(c);
    BilinearFormType bilinearForm = Laplacian(diffusion_.asParameter()) + Gradient(advection_.asParameter()) + Identity(reaction_.asParameter());
    pde_.setBilinearForm(bilinearForm);  
  };

  // getters
  DMatrix<double> get_quadrature_nodes() const { return pde_.integrator().quadratureNodes(domain_); };
  DMatrix<double> get_dofs_coordinates() const { return domain_.dofCoords(); };
  const PDE<M,N,R, BilinearFormType, DMatrix<double>>& pde() { return pde_; }
  
  // compile time informations
  typedef PDE<M,N,R, BilinearFormType, DMatrix<double>> PDEType;
};

// define 2D costant coefficient PDE regularization.
typedef RegularizingPDE<2,2,1, SpaceVaryingPDE<2>> SpaceVarying_2D_Order1;

#endif // __COMMON_H__
