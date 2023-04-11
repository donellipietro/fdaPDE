#ifndef __FPLSR_H__
#define __FPLSR_H__

#include <Eigen/SVD>
#include "../../core/utils/Symbols.h"
#include "../../calibration/GCV.h"
#include "../../calibration/KFoldCV.h"
using fdaPDE::calibration::GCV;
using fdaPDE::calibration::KFoldCV;
using fdaPDE::calibration::StochasticEDF;
#include "FPIREM.h"
#include "PCScoreCV.h"
using fdaPDE::models::PCScoreCV;
#include "../../core/OPT/optimizers/GridOptimizer.h"
using fdaPDE::core::OPT::GridOptimizer;
#include "FunctionalRegressionBase.h"
using fdaPDE::models::FunctionalRegressionBase;

namespace fdaPDE
{
    namespace models
    {

        struct fixed_lambda
        {
        };
        struct gcv_lambda_selection
        {
        };
        struct kcv_lambda_selection
        {
        };

        // base class for any FPCA model
        template <typename PDE, typename RegularizationType, Sampling SamplingDesign, typename lambda_selection_strategy>
        class FPLSR : public FunctionalRegressionBase<FPLSR<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>>
        {
            // compile time checks
            static_assert(std::is_base_of<PDEBase, PDE>::value);

        private:
            typedef FPLSR<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy> ModelType;
            typedef FunctionalRegressionBase<ModelType> Base;
            typedef typename FPIREM<ModelType>::SmootherType SmootherType;

            // parameters used as stopping criterion by FPIREM algorithm
            std::size_t max_iter_ = 20; // maximum number of iterations before forced stop
            double tol_ = 1e-2;         // tolerance on |Jnew - Jold| used as convergence criterion

            // default number of latent components
            std::size_t H_ = 3;

            // problem solution
            DMatrix<double> W_; // directions in X-spce
            DMatrix<double> V_; // directions in Y-space
            DMatrix<double> T_; // latent components
            DMatrix<double> C_; // X components
            DMatrix<double> D_; // Y components

            // residuals
            BlockFrame<double, int> df_residuals_; // (OBSERVATIONS_BLK): F_Y, (DESIGN_MATRIX_BLK): F_X
            DMatrix<double> &F_Y() { return df_residuals_.template get<double>(OBSERVATIONS_BLK); }
            DMatrix<double> &F_X() { return df_residuals_.template get<double>(DESIGN_MATRIX_BLK); }

            // tag dispatched private methods for computation of PCs
            void solve_(fixed_lambda);
            void solve_(gcv_lambda_selection);
            void solve_(kcv_lambda_selection);

            // required to support \lambda parameter selection
            std::vector<SVector<model_traits<SmootherType>::n_lambda>> lambda_vect_;

        public:
            using Base::lambda;

            // constructor
            FPLSR() = default;
            // space-only constructor
            template <typename U = RegularizationType,
                      typename std::enable_if<std::is_same<U, SpaceOnly>::value, int>::type = 0>
            FPLSR(const PDE &pde) : Base(pde){};
            // space-time constructor
            template <typename U = RegularizationType,
                      typename std::enable_if<!std::is_same<U, SpaceOnly>::value, int>::type = 0>
            FPLSR(const PDE &pde, const DVector<double> &time) : Base(pde, time){};

            void init_model();    // initialize the model
            virtual void solve(); // compute latent components

            // getters
            const DMatrix<double> &F_Y() const { return df_residuals_.template get<double>(OBSERVATIONS_BLK); }
            const DMatrix<double> &F_X() const { return df_residuals_.template get<double>(DESIGN_MATRIX_BLK); }
            const DMatrix<double> &W() const { return W_; }
            const DMatrix<double> &T() const { return T_; }
            const DMatrix<double> &C() const { return C_; }
            const DMatrix<double> &D() const { return D_; }

            // setters
            void setTolerance(double tol) { tol_ = tol; }
            void setMaxIterations(std::size_t max_iter) { max_iter_ = max_iter; }
            void setH(std::size_t H) { H_ = H; }
            // accepts a collection of \lambda parameters if a not fixed_lambda method is selected
            void setLambda(const std::vector<SVector<model_traits<SmootherType>::n_lambda>> &lambda_vect)
            {
                lambda_vect_ = lambda_vect;
            }

            virtual ~FPLSR() = default;
        };
        template <typename PDE_, typename RegularizationType_,
                  Sampling SamplingDesign_, typename lambda_selection_strategy>
        struct model_traits<FPLSR<PDE_, RegularizationType_, SamplingDesign_, lambda_selection_strategy>>
        {
            typedef PDE_ PDE;
            typedef RegularizationType_ RegularizationType;
            static constexpr Sampling sampling = SamplingDesign_;
            static constexpr SolverType solver = SolverType::Monolithic;
            static constexpr int n_lambda = n_smoothing_parameters<RegularizationType>::value;
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
