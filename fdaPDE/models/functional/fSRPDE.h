#ifndef __FSRPDE_H__
#define __FSRPDE_H__

#include "../../core/utils/Symbols.h"
#include "../regression/SRPDE.h"
using fdaPDE::models::SRPDE;

namespace fdaPDE
{
    namespace models
    {

        // wrapper to apply SRPDE to functional data
        template <typename PDE, typename SamplingDesign>
        class FSRPDE
        {
            // compile time checks
            static_assert(std::is_base_of<PDEBase, PDE>::value);

        private:
            typedef FSRPDE<PDE, SamplingDesign> ModelType;
            typedef SRPDE<PDE, SamplingDesign> SmootherType;

            // solver
            SmootherType solver_;

            // problem dimensions
            std::size_t S_; // number of observation points
            std::size_t N_; // number of samples
            std::size_t K_; // number of basis

            // data
            BlockFrame<double, int> df_solver_;
            double b_norm_;

            // problem solution
            DMatrix<double> f_;

        public:
            // constructor
            FSRPDE() = default;
            FSRPDE(const PDE &pde)
            {
                // std::cout << "initialization fSRPDE" << std::endl;

                setPDE(pde);

                // std::cout << "initialization fSRPDE" << std::endl;
            };

            // getters
            DMatrix<double> f() const { return f_; }
            DMatrix<double> fitted() const { return f_ * solver_.PsiTD(); }

            // setters
            void setPDE(const PDE &pde)
            {
                // std::cout << "set_pde" << std::endl;

                solver_.setPDE(pde);
                solver_.init_pde();

                // number of mesh nodes
                K_ = solver_.n_basis();

                // reserve space for solution
                f_.resize(1, K_);

                // std::cout << "set_pde" << std::endl;
            }

            void setLambdaS(double lambda)
            {
                // std::cout << "set_lambda" << std::endl;

                solver_.setLambdaS(lambda);

                // std::cout << "set_lambda" << std::endl;
            }

            void set_spatial_locations(const DMatrix<double> &locs)
            {
                // std::cout << "set_locations" << std::endl;

                solver_.set_spatial_locations(locs);

                // std::cout << "set_locations" << std::endl;
            }

            void setData(const BlockFrame<double, int> &df)
            {
                // std::cout << "set_data" << std::endl;

                // unpack data
                const auto &X = df.get<double>(OBSERVATIONS_BLK);
                const auto &b = df.get<double>(DESIGN_MATRIX_BLK);

                setData(X, b);

                // std::cout << "set_data" << std::endl;
            }

            void setData(const DMatrix<double> &X)
            {
                // std::cout << "set_data" << std::endl;

                // dimensions
                N_ = X.rows();
                S_ = X.cols();

                // covariates norm (in this case b is assumed to be a vector of ones)
                b_norm_ = std::sqrt(N_);

                // solver's data
                df_solver_.insert<double>(OBSERVATIONS_BLK, X.colwise().sum().transpose() / b_norm_);

                // std::cout << "set_data" << std::endl;
            }

            void setData(const DMatrix<double> &X, const DVector<double> &b)
            {

                // std::cout << "set_data" << std::endl;

                // dimensions
                N_ = X.rows();
                S_ = X.cols();

                // covariates norm
                b_norm_ = b.norm();

                // solver's data
                df_solver_.insert<double>(OBSERVATIONS_BLK, X.transpose() * b / b_norm_);

                // std::cout << "set_data" << std::endl;
            }

            void init()
            {
                // std::cout << "init" << std::endl;

                // initialization of the solver
                solver_.setData(df_solver_);
                // std::cout << "regularization" << std::endl;
                solver_.init_regularization();
                // std::cout << "regularization" << std::endl;
                // std::cout << "sampling" << std::endl;
                solver_.init_sampling();
                // std::cout << "sampling" << std::endl;
                // std::cout << "init model" << std::endl;
                solver_.init_model();
                // std::cout << "init model" << std::endl;

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

            DMatrix<double> compute(const DMatrix<double> &X, const DVector<double> &b)
            {
                // std::cout << "compute" << std::endl;

                setData(X, b);
                init();
                solve();
                return f_;

                // std::cout << "compute" << std::endl;
            }

            DMatrix<double> compute(const DMatrix<double> &X)
            {
                // std::cout << "compute" << std::endl;

                setData(X);
                init();
                solve();

                // std::cout << "compute" << std::endl;

                return f_;
            }
        };

        template <typename PDE_, typename SamplingDesign_>
        struct model_traits<FSRPDE<PDE_, SamplingDesign_>>
        {
            typedef PDE_ PDE;
            typedef SpaceOnly regularization;
            typedef SamplingDesign_ sampling;
            typedef MonolithicSolver solver;
            static constexpr int n_lambda = 1;
        };

        template <typename Model>
        struct is_fsrpde
        {
            static constexpr bool value = is_instance_of<Model, FSRPDE>::value;
        };

    }
}

#endif // __FSRPDE_H__
