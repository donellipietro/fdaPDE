#ifndef __FSRPDE_H__
#define __FSRPDE_H__

#include <Eigen/SVD>
#include "../../core/utils/Symbols.h"
#include "../regression/SRPDE.h"
using fdaPDE::models::SRPDE;
// #include "../../calibration/GCV.h"
// #include "../../calibration/KFoldCV.h"
// using fdaPDE::calibration::GCV;
// using fdaPDE::calibration::KFoldCV;
// using fdaPDE::calibration::StochasticEDF;
// #include "../../core/OPT/optimizers/GridOptimizer.h"
// using fdaPDE::core::OPT::GridOptimizer;

namespace fdaPDE
{
    namespace models
    {

        // wrapper to apply SRPDE to functional data
        template <typename PDE, Sampling SamplingDesign>
        class FSRPDE
        {
            // compile time checks
            static_assert(std::is_base_of<PDEBase, PDE>::value);

        private:
            typedef FSRPDE<PDE, SamplingDesign> ModelType;
            typedef SRPDE<PDE, SamplingDesign> SmootherType;

            // solver
            SmootherType solver_;

            // data
            BlockFrame<double, int> df_;
            BlockFrame<double, int> df_solver_;
            double b_norm_;
            DMatrix<double> locs_;

            // problem solution
            DMatrix<double> f_;

        public:
            // constructor
            FSRPDE() = default;
            FSRPDE(const PDE &pde)
            {
                // std::cout << "initialization" << std::endl;

                setPDE(pde);

                // std::cout << "initialization" << std::endl;
            };

            // getters
            DMatrix<double> f() const { return f_; }

            // setters
            void setPDE(const PDE &pde)
            {
                // std::cout << "set_pde" << std::endl;

                solver_.setPDE(pde);
                solver_.init_pde();

                // std::cout << "set_pde" << std::endl;
            }
            void setLambdaS(double lambda)
            {
                // std::cout << "set_lambda" << std::endl;

                solver_.setLambdaS(lambda);

                // std::cout << "set_lambda" << std::endl;
            }
            void setLocations(const DMatrix<double> &locs)
            {
                // std::cout << "set_locations" << std::endl;

                locs_ = locs;

                // std::cout << "set_locations" << std::endl;
            }
            void setData(const BlockFrame<double, int> &df)
            {
                // std::cout << "set_data" << std::endl;

                df_ = df;
                b_norm_ = df_.get<double>(DESIGN_MATRIX_BLK).norm();

                // std::cout << "set_data" << std::endl;
            }
            void init()
            {
                // std::cout << "init" << std::endl;

                // solver's data
                const auto &X = df_.get<double>(OBSERVATIONS_BLK);
                const auto &b = df_.get<double>(DESIGN_MATRIX_BLK);
                df_solver_.insert<double>(OBSERVATIONS_BLK, X.transpose() * b / b_norm_);
                if constexpr (is_sampling_pointwise_at_locs<SmootherType>::value)
                    df_solver_.insert(SPACE_LOCATIONS_BLK, locs_);

                // initialization of the solver
                solver_.setData(df_solver_);
                solver_.init_regularization();
                solver_.init_sampling();
                solver_.init_model();

                // reserve space for solution
                f_.resize(1, X.cols());

                // std::cout << "init" << std::endl;
            }

            // methods
            void solve()
            {
                // std::cout << "solve" << std::endl;

                solver_.solve();
                f_ = solver_.f().transpose() / (b_norm_);

                // std::cout << "solve" << std::endl;
            }
        };

        template <typename PDE_, Sampling SamplingDesign>
        struct model_traits<FSRPDE<PDE_, SamplingDesign>>
        {
            typedef PDE_ PDE;
            typedef SpaceOnly RegularizationType;
            static constexpr Sampling sampling = SamplingDesign;
            static constexpr SolverType solver = SolverType::Monolithic;
            static constexpr int n_lambda = 1;
        };

    }
}

#endif // __FSRPDE_H__
