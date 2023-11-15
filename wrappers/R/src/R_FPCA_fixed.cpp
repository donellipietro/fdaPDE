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
class R_FPCA_fixed
{
protected:
    typedef RegularizingPDE RegularizingPDE_;
    RegularizingPDE_ regularization_;

    FPCA<typename RegularizingPDE_::PDEType, fdaPDE::models::SpaceOnly, S, fdaPDE::models::fixed_lambda> model_;
    BlockFrame<double, int> df_;

public:
    // Constructor
    R_FPCA_fixed(const RegularizingPDE_ &regularization)
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
    std::vector<double> lambda_opt() const
    {
        std::vector<double> lambda;
        for (const auto v : model_.lambda_opt())
            lambda.push_back(v[0]);
        return lambda;
    }

    // Setters
    void set_npc(std::size_t n) { model_.set_npc(n); }
    void set_lambda_s(std::vector<double> lambdas)
    {
        std::vector<SVector<1>> l_;
        for (auto v : lambdas)
            l_.push_back(SVector<1>(v));
        model_.set_lambda_s_comp(l_);
        model_.setLambdaS(lambdas.front());
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
typedef R_FPCA_fixed<Laplacian_2D_Order1, fdaPDE::models::GeoStatMeshNodes> FPCA_Laplacian_2D_GeoStatNodes_fixed;
RCPP_MODULE(FPCA_Laplacian_2D_GeoStatNodes_fixed)
{
    Rcpp::class_<FPCA_Laplacian_2D_GeoStatNodes_fixed>("FPCA_Laplacian_2D_GeoStatNodes_fixed")
        .constructor<Laplacian_2D_Order1>()
        // Initializations
        .method("init", &FPCA_Laplacian_2D_GeoStatNodes_fixed::init)
        .method("init_regularization", &FPCA_Laplacian_2D_GeoStatNodes_fixed::init_regularization)
        .method("init_pde", &FPCA_Laplacian_2D_GeoStatNodes_fixed::init_pde)
        // Getters
        .method("W", &FPCA_Laplacian_2D_GeoStatNodes_fixed::W)
        .method("loadings", &FPCA_Laplacian_2D_GeoStatNodes_fixed::loadings)
        .method("scores", &FPCA_Laplacian_2D_GeoStatNodes_fixed::scores)
        .method("R0", &FPCA_Laplacian_2D_GeoStatNodes_fixed::R0)
        .method("Psi", &FPCA_Laplacian_2D_GeoStatNodes_fixed::Psi)
        .method("lambda_opt", &FPCA_Laplacian_2D_GeoStatNodes_fixed::lambda_opt)
        // Setters
        .method("set_lambda_s", &FPCA_Laplacian_2D_GeoStatNodes_fixed::set_lambda_s)
        .method("set_npc", &FPCA_Laplacian_2D_GeoStatNodes_fixed::set_npc)
        .method("set_observations", &FPCA_Laplacian_2D_GeoStatNodes_fixed::set_observations)
        // Solve method
        .method("solve", &FPCA_Laplacian_2D_GeoStatNodes_fixed::solve);
}

// Locations != Nodes
typedef R_FPCA_fixed<Laplacian_2D_Order1, fdaPDE::models::GeoStatLocations> FPCA_Laplacian_2D_GeoStatLocations_fixed;
RCPP_MODULE(FPCA_Laplacian_2D_GeoStatLocations_fixed)
{
    Rcpp::class_<FPCA_Laplacian_2D_GeoStatLocations_fixed>("FPCA_Laplacian_2D_GeoStatLocations_fixed")
        .constructor<Laplacian_2D_Order1>()
        // Initializations
        .method("init_regularization", &FPCA_Laplacian_2D_GeoStatLocations_fixed::init_regularization)
        .method("init_pde", &FPCA_Laplacian_2D_GeoStatLocations_fixed::init_pde)
        .method("init", &FPCA_Laplacian_2D_GeoStatLocations_fixed::init)
        // Getters
        .method("W", &FPCA_Laplacian_2D_GeoStatLocations_fixed::W)
        .method("loadings", &FPCA_Laplacian_2D_GeoStatLocations_fixed::loadings)
        .method("scores", &FPCA_Laplacian_2D_GeoStatLocations_fixed::scores)
        .method("R0", &FPCA_Laplacian_2D_GeoStatLocations_fixed::R0)
        .method("Psi", &FPCA_Laplacian_2D_GeoStatLocations_fixed::Psi)
        .method("lambda_opt", &FPCA_Laplacian_2D_GeoStatLocations_fixed::lambda_opt)
        // Setters
        .method("set_lambda_s", &FPCA_Laplacian_2D_GeoStatLocations_fixed::set_lambda_s)
        .method("set_npc", &FPCA_Laplacian_2D_GeoStatLocations_fixed::set_npc)
        .method("set_locations", &FPCA_Laplacian_2D_GeoStatLocations_fixed::set_locations)
        .method("set_observations", &FPCA_Laplacian_2D_GeoStatLocations_fixed::set_observations)
        // Solve method
        .method("solve", &FPCA_Laplacian_2D_GeoStatLocations_fixed::solve);
}

// 3D wrapper
typedef R_FPCA_fixed<Laplacian_3D_Order1, fdaPDE::models::GeoStatMeshNodes> FPCA_Laplacian_3D_GeoStatNodes_fixed;
RCPP_MODULE(FPCA_Laplacian_3D_GeoStatNodes_fixed)
{
    Rcpp::class_<FPCA_Laplacian_3D_GeoStatNodes_fixed>("FPCA_Laplacian_3D_GeoStatNodes_fixed")
        .constructor<Laplacian_3D_Order1>()
        // Initializations
        .method("init_regularization", &FPCA_Laplacian_3D_GeoStatNodes_fixed::init_regularization)
        .method("init_pde", &FPCA_Laplacian_3D_GeoStatNodes_fixed::init_pde)
        // Getters
        .method("W", &FPCA_Laplacian_3D_GeoStatNodes_fixed::W)
        .method("loadings", &FPCA_Laplacian_3D_GeoStatNodes_fixed::loadings)
        .method("scores", &FPCA_Laplacian_3D_GeoStatNodes_fixed::scores)
        .method("R0", &FPCA_Laplacian_3D_GeoStatNodes_fixed::R0)
        .method("lambda_opt", &FPCA_Laplacian_3D_GeoStatNodes_fixed::lambda_opt)
        // Setters
        .method("set_lambda_s", &FPCA_Laplacian_3D_GeoStatNodes_fixed::set_lambda_s)
        .method("set_observations", &FPCA_Laplacian_3D_GeoStatNodes_fixed::set_observations)
        // Solve method
        .method("solve", &FPCA_Laplacian_3D_GeoStatNodes_fixed::solve);
}
