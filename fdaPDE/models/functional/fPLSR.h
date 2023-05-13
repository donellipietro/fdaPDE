#ifndef __FPLSR_H__
#define __FPLSR_H__

#include <Eigen/SVD>
#include "../../core/utils/Symbols.h"
#include "../../core/OPT/optimizers/GridOptimizer.h"
using fdaPDE::core::OPT::GridOptimizer;
#include "../../calibration/GCV.h"
using fdaPDE::calibration::GCV;
#include "FunctionalBase.h"
using fdaPDE::models::FunctionalBase;
#include "ProfilingEstimation.h"
#include "FunctionalRegressionBase.h"
using fdaPDE::models::FunctionalRegressionBase;

namespace fdaPDE
{
    namespace models
    {

        // base class for any FPCA model
        template <typename PDE, typename RegularizationType, typename SamplingDesign, typename lambda_selection_strategy>
        class FPLSR : public FunctionalRegressionBase<FPLSR<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>>
        {
            // compile time checks
            static_assert(std::is_base_of<PDEBase, PDE>::value);

        private:
            typedef FPLSR<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy> ModelType;
            typedef FunctionalRegressionBase<ModelType> Base;

            // parameters used as stopping criterion by FPIREM algorithm
            std::size_t max_iter_ = 20; // maximum number of iterations before forced stop
            double tol_ = 1e-2;         // tolerance on |Jnew - Jold| used as convergence criterion

            // default number of latent components
            std::size_t H_ = 3;

            // spatial matrices
            SpMatrix<double> PsiTPsi_{};
            fdaPDE::SparseLU<SpMatrix<double>> invPsiTPsi_;

            // problem solution
            DMatrix<double> W_; // directions in X-spce
            DMatrix<double> V_; // directions in Y-space
            DMatrix<double> T_; // latent components
            DMatrix<double> C_; // X components
            DMatrix<double> D_; // Y components

            // residuals
            BlockFrame<double, int> df_residuals_; // (OBSERVATIONS_BLK): F, (DESIGN_MATRIX_BLK): E
            DMatrix<double> &F() { return df_residuals_.template get<double>(OBSERVATIONS_BLK); }
            DMatrix<double> &E() { return df_residuals_.template get<double>(DESIGN_MATRIX_BLK); }

            // tag dispatched private methods for computation of PCs
            void solve_(fixed_lambda);
            void solve_(gcv_lambda_selection);
            void solve_(kcv_lambda_selection);

        public:
            IMPORT_MODEL_SYMBOLS;
            using Base::lambda;
            using Base::lambdas;

            // constructor
            FPLSR() = default;
            // space-only constructor
            template <typename U = RegularizationType,
                      typename std::enable_if<std::is_same<U, SpaceOnly>::value, int>::type = 0>
            FPLSR(const PDE &pde) : Base(pde){
                                        // std::cout << "initialization fPLSR" << std::endl;
                                    };
            // space-time constructor
            template <typename U = RegularizationType,
                      typename std::enable_if<!std::is_same<U, SpaceOnly>::value, int>::type = 0>
            FPLSR(const PDE &pde, const DVector<double> &time) : Base(pde, time){
                                                                     // std::cout << "initialization fPLSR" << std::endl;
                                                                 };

            void init_model();    // initialize the model
            virtual void solve(); // compute latent components

            // getters
            const DMatrix<double> &F() const { return df_residuals_.template get<double>(OBSERVATIONS_BLK); }
            const DMatrix<double> &E() const { return df_residuals_.template get<double>(DESIGN_MATRIX_BLK); }
            const DMatrix<double> &W() const { return W_; }
            const DMatrix<double> &T() const { return T_; }
            const DMatrix<double> &C() const { return C_; }
            const DMatrix<double> &D() const { return D_; }
            const SpMatrix<double> &PsiTPsi() const { return PsiTPsi_; }
            const fdaPDE::SparseLU<SpMatrix<double>> &invPsiTPsi() const { return invPsiTPsi_; }

            // setters
            void set_tolerance(double tol) { tol_ = tol; }
            void set_max_iterations(std::size_t max_iter) { max_iter_ = max_iter; }
            void set_H(std::size_t H) { H_ = H; }

            // methods
            DMatrix<double> reconstructed_field() const
            {
                return (T() * C().transpose()).rowwise() + this->X_mean().transpose();
            }

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
