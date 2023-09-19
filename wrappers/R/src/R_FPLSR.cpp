// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include <thread>
#include <fdaPDE/core/utils/Symbols.h>
#include <fdaPDE/models/functional/fPLSR.h>
using fdaPDE::models::FPLSR;
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

// wrapper for fPLSR Closed Solution module
template <typename RegularizingPDE, typename S>
class R_FPLSR
{
protected:
    typedef RegularizingPDE RegularizingPDE_;
    RegularizingPDE_ regularization_;

    FPLSR<typename RegularizingPDE_::PDEType, fdaPDE::models::SpaceOnly, S, fdaPDE::models::fixed_lambda> model_;
    BlockFrame<double, int> df_;

public:
    // Constructor
    R_FPLSR(const RegularizingPDE_ &regularization)
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

    // Getters model base
    SpMatrix<double> R0() const { return model_.R0(); }
    SpMatrix<double> Psi() { return model_.Psi(fdaPDE::models::not_nan()); }

    // Getters functional regression
    DMatrix<double> B() const { return model_.B(); }
    DMatrix<double> Y_mean() const { return model_.Y_mean(); }
    DMatrix<double> X_mean() const { return model_.X_mean(); }
    DMatrix<double> reconstructed_field() const { return model_.reconstructed_field(); }
    DMatrix<double> fitted() const { return model_.fitted(); }
    DMatrix<double> predict(const DVector<double> &covs) const { return model_.predict(covs); }

    // Getters fPLSR
    DMatrix<double> F() const { return model_.F(); }
    DMatrix<double> E() const { return model_.E(); }
    DMatrix<double> W() const { return model_.W(); }
    DMatrix<double> V() const { return model_.V(); }
    DMatrix<double> T() const { return model_.T(); }
    DMatrix<double> C() const { return model_.C(); }
    DMatrix<double> D() const { return model_.D(); }

    // Setters controls
    void set_tolerance(double tol) { model_.set_tolerance(tol); }
    void set_max_iterations(std::size_t max_iter) { model_.set_max_iterations(max_iter); }
    void set_H(std::size_t H) { model_.set_H(H); }

    // Setters options
    void set_verbose(bool verbose) { model_.set_verbose(verbose); }
    void set_full_functional(bool full_functional) { model_.set_full_functional(full_functional); }
    void set_smoothing0(bool smoothing) { model_.set_smoothing(smoothing); }
    void set_smoothing1(bool smoothing, double lambda_smoothing) { model_.set_smoothing(smoothing, lambda_smoothing); }
    void set_smoothing2(bool smoothing_initialization, bool smoothing_regression) { model_.set_smoothing(smoothing_initialization, smoothing_regression); }
    void set_smoothing3(bool smoothing_initialization, bool smoothing_regression, double lambda_smoothing_initialization, double lambda_smoothing_regression) { model_.set_smoothing(smoothing_initialization, smoothing_regression, lambda_smoothing_initialization, lambda_smoothing_regression); }

    // Setters data and parameters
    void set_lambda_s(double lambdaS) { model_.setLambdaS(lambdaS); }
    void set_data(const DMatrix<double> &Y, const DMatrix<double> &X)
    {
        df_.template insert<double>(OBSERVATIONS_BLK, Y);
        df_.template insert<double>(DESIGN_MATRIX_BLK, X);
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

// RCPP modules for R_FPLSR

// Laplacian_2D_Order1, locations == nodes
typedef R_FPLSR<Laplacian_2D_Order1, fdaPDE::models::GeoStatMeshNodes> FPLSR_Laplacian_2D_GeoStatNodes;
RCPP_MODULE(FPLSR_Laplacian_2D_GeoStatNodes)
{
    Rcpp::class_<FPLSR_Laplacian_2D_GeoStatNodes>("FPLSR_Laplacian_2D_GeoStatNodes")
        .constructor<Laplacian_2D_Order1>()
        // Initializations
        .method("init_pde", &FPLSR_Laplacian_2D_GeoStatNodes::init_pde)
        .method("init", &FPLSR_Laplacian_2D_GeoStatNodes::init)
        .method("init_regularization", &FPLSR_Laplacian_2D_GeoStatNodes::init_regularization)
        // Getters model base
        .method("R0", &FPLSR_Laplacian_2D_GeoStatNodes::R0)
        .method("Psi", &FPLSR_Laplacian_2D_GeoStatNodes::Psi)
        // Getters functional regression
        .method("B", &FPLSR_Laplacian_2D_GeoStatNodes::B)
        .method("Y_mean", &FPLSR_Laplacian_2D_GeoStatNodes::Y_mean)
        .method("X_mean", &FPLSR_Laplacian_2D_GeoStatNodes::X_mean)
        .method("reconstructed_field", &FPLSR_Laplacian_2D_GeoStatNodes::reconstructed_field)
        .method("fitted", &FPLSR_Laplacian_2D_GeoStatNodes::fitted)
        .method("predict", &FPLSR_Laplacian_2D_GeoStatNodes::predict)
        // Getters fPLSR
        .method("F", &FPLSR_Laplacian_2D_GeoStatNodes::F)
        .method("E", &FPLSR_Laplacian_2D_GeoStatNodes::E)
        .method("W", &FPLSR_Laplacian_2D_GeoStatNodes::W)
        .method("V", &FPLSR_Laplacian_2D_GeoStatNodes::V)
        .method("T", &FPLSR_Laplacian_2D_GeoStatNodes::T)
        .method("C", &FPLSR_Laplacian_2D_GeoStatNodes::C)
        .method("D", &FPLSR_Laplacian_2D_GeoStatNodes::D)
        // Setters controls
        .method("set_tollerance", &FPLSR_Laplacian_2D_GeoStatNodes::set_tolerance)
        .method("set_max_iterations", &FPLSR_Laplacian_2D_GeoStatNodes::set_max_iterations)
        .method("set_H", &FPLSR_Laplacian_2D_GeoStatNodes::set_H)
        // Setters options
        .method("set_verbose", &FPLSR_Laplacian_2D_GeoStatNodes::set_verbose)
        .method("set_full_functional", &FPLSR_Laplacian_2D_GeoStatNodes::set_full_functional)
        .method("set_smoothing0", &FPLSR_Laplacian_2D_GeoStatNodes::set_smoothing0)
        .method("set_smoothing1", &FPLSR_Laplacian_2D_GeoStatNodes::set_smoothing1)
        .method("set_smoothing2", &FPLSR_Laplacian_2D_GeoStatNodes::set_smoothing2)
        .method("set_smoothing3", &FPLSR_Laplacian_2D_GeoStatNodes::set_smoothing3)
        // Setters data and parameters
        .method("set_lambda_s", &FPLSR_Laplacian_2D_GeoStatNodes::set_lambda_s)
        .method("set_data", &FPLSR_Laplacian_2D_GeoStatNodes::set_data)
        .method("set_locations", &FPLSR_Laplacian_2D_GeoStatNodes::set_locations)
        // Solve method
        .method("solve", &FPLSR_Laplacian_2D_GeoStatNodes::solve);
}

// Laplacian_2D_Order1, locations != nodes
typedef R_FPLSR<Laplacian_2D_Order1, fdaPDE::models::GeoStatLocations> FPLSR_Laplacian_2D_GeoStatLocations;
RCPP_MODULE(FPLSR_Laplacian_2D_GeoStatLocations)
{
    Rcpp::class_<FPLSR_Laplacian_2D_GeoStatLocations>("FPLSR_Laplacian_2D_GeoStatLocations")
        .constructor<Laplacian_2D_Order1>()
        // Initializations
        .method("init_pde", &FPLSR_Laplacian_2D_GeoStatLocations::init_pde)
        .method("init", &FPLSR_Laplacian_2D_GeoStatLocations::init)
        .method("init_regularization", &FPLSR_Laplacian_2D_GeoStatLocations::init_regularization)
        // Getters model base
        .method("R0", &FPLSR_Laplacian_2D_GeoStatLocations::R0)
        .method("Psi", &FPLSR_Laplacian_2D_GeoStatLocations::Psi)
        // Getters functional regression
        .method("B", &FPLSR_Laplacian_2D_GeoStatLocations::B)
        .method("Y_mean", &FPLSR_Laplacian_2D_GeoStatLocations::Y_mean)
        .method("X_mean", &FPLSR_Laplacian_2D_GeoStatLocations::X_mean)
        .method("reconstructed_field", &FPLSR_Laplacian_2D_GeoStatLocations::reconstructed_field)
        .method("fitted", &FPLSR_Laplacian_2D_GeoStatLocations::fitted)
        .method("predict", &FPLSR_Laplacian_2D_GeoStatLocations::predict)
        // Getters fPLSR
        .method("F", &FPLSR_Laplacian_2D_GeoStatLocations::F)
        .method("E", &FPLSR_Laplacian_2D_GeoStatLocations::E)
        .method("W", &FPLSR_Laplacian_2D_GeoStatLocations::W)
        .method("V", &FPLSR_Laplacian_2D_GeoStatLocations::V)
        .method("T", &FPLSR_Laplacian_2D_GeoStatLocations::T)
        .method("C", &FPLSR_Laplacian_2D_GeoStatLocations::C)
        .method("D", &FPLSR_Laplacian_2D_GeoStatLocations::D)
        // Setters controls
        .method("set_tollerance", &FPLSR_Laplacian_2D_GeoStatLocations::set_tolerance)
        .method("set_max_iterations", &FPLSR_Laplacian_2D_GeoStatLocations::set_max_iterations)
        .method("set_H", &FPLSR_Laplacian_2D_GeoStatLocations::set_H)
        // Setters options
        .method("set_verbose", &FPLSR_Laplacian_2D_GeoStatLocations::set_verbose)
        .method("set_full_functional", &FPLSR_Laplacian_2D_GeoStatLocations::set_full_functional)
        .method("set_smoothing0", &FPLSR_Laplacian_2D_GeoStatLocations::set_smoothing0)
        .method("set_smoothing1", &FPLSR_Laplacian_2D_GeoStatLocations::set_smoothing1)
        .method("set_smoothing2", &FPLSR_Laplacian_2D_GeoStatLocations::set_smoothing2)
        .method("set_smoothing3", &FPLSR_Laplacian_2D_GeoStatLocations::set_smoothing3)
        // Setters data and parameters
        .method("set_lambda_s", &FPLSR_Laplacian_2D_GeoStatLocations::set_lambda_s)
        .method("set_data", &FPLSR_Laplacian_2D_GeoStatLocations::set_data)
        .method("set_locations", &FPLSR_Laplacian_2D_GeoStatLocations::set_locations)
        // Solve method
        .method("solve", &FPLSR_Laplacian_2D_GeoStatLocations::solve);
}