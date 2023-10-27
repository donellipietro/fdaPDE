#ifndef __FPLSR_H__
#define __FPLSR_H__

#include <Eigen/SVD>
#include "../../core/utils/Symbols.h"
#include "../../core/OPT/optimizers/GridOptimizer.h"
using fdaPDE::core::OPT::GridOptimizer;
#include "../../calibration/GCV.h"
#include "../../calibration/KFoldCV.h"
using fdaPDE::calibration::GCV;
using fdaPDE::calibration::KFoldCV;
#include "FunctionalBase.h"
using fdaPDE::models::FunctionalBase;
#include "ProfilingEstimation.h"
#include "FunctionalRegressionBase.h"
using fdaPDE::models::FunctionalRegressionBase;

namespace fdaPDE
{
    namespace models
    {

        template <typename PDE, typename RegularizationType, typename SamplingDesign, typename lambda_selection_strategy>
        class FPLSR : public FunctionalRegressionBase<FPLSR<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>>
        {
            // Compile time checks
            static_assert(std::is_base_of<PDEBase, PDE>::value);

        private:
            typedef FPLSR<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy> ModelType;
            typedef FunctionalRegressionBase<ModelType> Base;

            // Parameters used as stopping criterion by ProfilingEstimation algorithm
            std::size_t max_iter_ = 20; // maximum number of iterations before forced stop
            double tol_ = 1e-2;         // tolerance on |Jnew - Jold| used as convergence criterion

            // Default number of latent components
            std::size_t H_ = 3;

            // Smoothing
            bool smoothing_regression_ = true;
            std::vector<SVector<1>> lambdas_smoothing_regression_ = {SVector<1>{1e-12}};
            std::vector<SVector<1>> lambda_smoothing_directions_ = {SVector<1>{std::numeric_limits<double>::quiet_NaN()}};
            std::vector<SVector<1>> lambda_smoothing_regression_ = {SVector<1>{std::numeric_limits<double>::quiet_NaN()}};

            // Spatial matrices
            SpMatrix<double> PsiTPsi_{};
            fdaPDE::SparseLU<SpMatrix<double>> invPsiTPsi_;

            // Problem solution
            DMatrix<double> W_; // directions in X-spce
            DMatrix<double> V_; // directions in Y-space
            DMatrix<double> T_; // latent components
            DMatrix<double> C_; // X components
            DMatrix<double> D_; // Y components

            // Residuals
            BlockFrame<double, int> df_residuals_; // (OBSERVATIONS_BLK): F, (DESIGN_MATRIX_BLK): E
            DMatrix<double> &F() { return df_residuals_.template get<double>(OBSERVATIONS_BLK); }
            DMatrix<double> &E() { return df_residuals_.template get<double>(DESIGN_MATRIX_BLK); }

            // Tag dispatched private methods for computation of PCs
            void solve_(fixed_lambda);
            void solve_(gcv_lambda_selection);
            void solve_(kcv_lambda_selection);

        public:
            IMPORT_MODEL_SYMBOLS;
            using Base::lambda;
            using Base::lambdas;

            // Constructor
            FPLSR() = default;
            // Space-only constructor
            template <typename U = RegularizationType,
                      typename std::enable_if<std::is_same<U, SpaceOnly>::value, int>::type = 0>
            FPLSR(const PDE &pde) : Base(pde){};
            // Space-time constructor
            /*
            template <typename U = RegularizationType,
                      typename std::enable_if<!std::is_same<U, SpaceOnly>::value, int>::type = 0>
            FPLSR(const PDE &pde, const DVector<double> &time) : Base(pde, time){};
            */

            void init_model();    // initialize the model
            virtual void solve(); // compute latent components

            // Getters
            const std::size_t H() const { return H_; }
            const DMatrix<double> &F() const { return df_residuals_.template get<double>(OBSERVATIONS_BLK); }
            const DMatrix<double> &E() const { return df_residuals_.template get<double>(DESIGN_MATRIX_BLK); }
            const DMatrix<double> &W() const { return W_; }
            const DMatrix<double> &V() const { return V_; }
            const DMatrix<double> &T() const { return T_; }
            const DMatrix<double> &C() const { return C_; }
            const DMatrix<double> &D() const { return D_; }
            const DMatrix<double> B(std::size_t h = 0) const
            {
                h = check_h(h);
                return compute_B(h);
            }
            const SpMatrix<double> &PsiTPsi() const { return PsiTPsi_; }
            const fdaPDE::SparseLU<SpMatrix<double>> &invPsiTPsi() const { return invPsiTPsi_; }
            std::size_t get_H() const { return H_; }
            std::vector<SVector<1>> get_lambda_directions() const { return lambda_smoothing_directions_; }
            std::vector<SVector<1>> get_lambda_regression() const { return lambda_smoothing_regression_; }
            SVector<1> get_lambda_initialization() const { return this->lambda_smoothing_initialization_; }

            // Setters
            void set_tolerance(double tol) { tol_ = tol; }
            void set_max_iterations(std::size_t max_iter) { max_iter_ = max_iter; }
            void set_H(std::size_t H) { H_ = H; }
            void set_full_functional(bool full_functional) { this->full_functional_ = full_functional; }
            void set_smoothing_regression(bool smoothing_regression)
            {
                smoothing_regression_ = smoothing_regression;
            }
            void set_smoothing_regression(bool smoothing_regression, std::vector<SVector<1>> lambdas_smoothing_regression)
            {
                smoothing_regression_ = smoothing_regression;
                lambdas_smoothing_regression_ = lambdas_smoothing_regression;
            }

            // Methods
            double f_norm(const DVector<double> &f) const;
            DMatrix<double> compute_B(std::size_t h = 0) const;
            DMatrix<double> fitted(std::size_t h = 0) const;
            std::size_t check_h(std::size_t h) const;
            DMatrix<double> reconstructed_field(std::size_t h = 0) const;

            virtual ~FPLSR() = default;
        };
        template <typename PDE_, typename RegularizationType_,
                  typename SamplingDesign_, typename lambda_selection_strategy>
        struct model_traits<FPLSR<PDE_, RegularizationType_, SamplingDesign_, lambda_selection_strategy>>
        {
            typedef PDE_ PDE;
            typedef fdaPDE::models::SpaceOnly regularization;
            typedef SamplingDesign_ sampling;
            typedef fdaPDE::models::MonolithicSolver solver;
            static constexpr int n_lambda = 1;
        };

        /*
        // specialization for separable regularization
        template <typename PDE_, Sampling SamplingDesign, typename lambda_selection_strategy>
        struct model_traits<FPLSR<PDE_, fdaPDE::models::SpaceTimeSeparable, SamplingDesign, lambda_selection_strategy>>
        {
            typedef PDE_ PDE;
            typedef fdaPDE::models::SpaceTimeSeparable RegularizationType;
            typedef SplineBasis<3> TimeBasis; // use cubic B-splines
            static constexpr Sampling sampling = SamplingDesign;
            static constexpr SolverType solver = SolverType::Monolithic;
            static constexpr int n_lambda = 2;
        };
        */

#include "fPLSR.tpp"

    }
}

#endif // __FPLSR_H__
