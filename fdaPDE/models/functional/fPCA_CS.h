#ifndef __FPCA_CS_H__
#define __FPCA_CS_H__

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
#include "RSVD.h"

namespace fdaPDE
{
    namespace models
    {

        struct fixed_lambda {};
        struct gcv_lambda_selection {};
        struct kcv_lambda_selection {};

        // base class for any FPCA_CS model
        template <typename PDE, typename RegularizationType, typename SamplingDesign, typename lambda_selection_strategy>
        class FPCA_CS : public FunctionalBase<FPCA_CS<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>>
        {
            // compile time checks
            static_assert(std::is_base_of<fdaPDE::core::FEM::PDEBase, PDE>::value);

        private:
            typedef FunctionalBase<FPCA_CS<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>> Base;
            std::size_t n_pc_ = 3; // default number of principal components

            // Intermediate steps matrices
            SpMatrix<double> PsiTPsi_{};
            fdaPDE::SparseLU<SpMatrix<double>> invPsiTPsi_;
            SpMatrix<double> P_{};

            // problem solution
            DMatrix<double> loadings_;
            DMatrix<double> scores_;

            // Options
            bool mass_lumping_ = false;

            // tag dispatched private methods for computation of PCs, ** to be removed **
            void solve_(fixed_lambda);
            void solve_(gcv_lambda_selection);
            void solve_(kcv_lambda_selection);

        public:
            IMPORT_MODEL_SYMBOLS;
            using Base::lambda;
            using Base::lambdas;
            using Base::nan_idxs;
            using Base::X;
            // constructor
            FPCA_CS() = default;
            // space-only constructor
            template <typename U = RegularizationType,
                      typename std::enable_if<std::is_same<U, SpaceOnly>::value, int>::type = 0>
            FPCA_CS(const PDE &pde) : Base(pde){};
            // space-time constructor
            template <typename U = RegularizationType,
                      typename std::enable_if<!std::is_same<U, SpaceOnly>::value, int>::type = 0>
            FPCA_CS(const PDE &pde, const DVector<double> &time) : Base(pde, time){};

            void init_model() { return; };
            virtual void solve(); // compute principal components

            // getters
            const DMatrix<double> &loadings() const { return loadings_; }
            const DMatrix<double> &scores() const { return scores_; }

            // setters
            void set_npc(std::size_t n_pc) { n_pc_ = n_pc; }
            void set_mass_lumping(bool mass_lumping) { mass_lumping_ = mass_lumping; }
        };
        template <typename PDE_, typename SamplingDesign_, typename lambda_selection_strategy>
        struct model_traits<FPCA_CS<PDE_, fdaPDE::models::SpaceOnly, SamplingDesign_, lambda_selection_strategy>>
        {
            typedef PDE_ PDE;
            typedef fdaPDE::models::SpaceOnly regularization;
            typedef SamplingDesign_ sampling;
            typedef fdaPDE::models::MonolithicSolver solver;
            static constexpr int n_lambda = 1;
        };
        // specialization for separable regularization
        template <typename PDE_, typename SamplingDesign_, typename lambda_selection_strategy>
        struct model_traits<FPCA_CS<PDE_, fdaPDE::models::SpaceTimeSeparable, SamplingDesign_, lambda_selection_strategy>>
        {
            typedef PDE_ PDE;
            typedef fdaPDE::models::SpaceTimeSeparable regularization;
            typedef SplineBasis<3> TimeBasis; // use cubic B-splines
            typedef SamplingDesign_ sampling;
            typedef fdaPDE::models::MonolithicSolver solver;
            static constexpr int n_lambda = 2;
        };

#include "fPCA_CS.tpp"

    }
}

#endif // __FPCA_CS_H__
