#ifndef __FRPDE_H__
#define __FRPDE_H__

#include "../../core/utils/Symbols.h"
#include "../regression/SRPDE.h"
using fdaPDE::models::SRPDE;

namespace fdaPDE
{
    namespace models
    {

        // Wrapper to apply SRPDE to functional data
        template <typename PDE, typename SamplingDesign>
        class FRPDE
        {
            // Compile time checks
            static_assert(std::is_base_of<PDEBase, PDE>::value);

        private:
            typedef FRPDE<PDE, SamplingDesign> ModelType;
            typedef SRPDE<PDE, SamplingDesign> SmootherType;

            // Solver
            SmootherType solver_;

            // Problem dimensions
            std::size_t S_; // number of observation points
            std::size_t N_; // number of samples
            std::size_t K_; // number of basis

            // Data
            BlockFrame<double, int> df_solver_;
            double b_norm_;

            // Problem solution
            DMatrix<double> f_;

            // Options
            bool verbose_ = false;

        public:
            // constructor
            FRPDE() = default;
            FRPDE(const PDE &pde)
            {
                setPDE(pde);
            };

            // getters
            DMatrix<double> f() const { return f_; }
            DMatrix<double> fitted() const { return f_ * solver_.PsiTD(); }
            double lambdaS() const { return solver_.lambdaS(); }

            // setters
            void setPDE(const PDE &pde)
            {
                // Set pde
                solver_.setPDE(pde);
                solver_.init_pde();

                // Number of mesh nodes
                K_ = solver_.n_basis();

                // Reserve space for solution
                f_.resize(1, K_);

                return;
            }

            void set_verbose(bool verbose) { verbose_ = verbose; }

            void setLambdaS(double lambda)
            {
                solver_.setLambdaS(lambda);

                return;
            }

            void set_spatial_locations(const DMatrix<double> &locs)
            {
                solver_.set_spatial_locations(locs);

                return;
            }

            void setData(const BlockFrame<double, int> &df)
            {
                // Unpack data
                const auto &X = df.get<double>(OBSERVATIONS_BLK);
                const auto &b = df.get<double>(DESIGN_MATRIX_BLK);

                // Set data
                setData(X, b);

                return;
            }

            void setData(const DMatrix<double> &X)
            {
                // Dimensions
                N_ = X.rows();
                S_ = X.cols();

                // Covariates norm (in this case b is assumed to be a vector of ones)
                b_norm_ = std::sqrt(N_);

                // Solver's data
                df_solver_.insert<double>(OBSERVATIONS_BLK, X.colwise().sum().transpose() / b_norm_);

                return;
            }

            void setData(const DMatrix<double> &X, const DVector<double> &b)
            {
                // Dimensions
                N_ = X.rows();
                S_ = X.cols();

                // Covariates norm
                b_norm_ = b.norm();

                // Solver's data
                df_solver_.insert<double>(OBSERVATIONS_BLK, X.transpose() * b / b_norm_);

                return;
            }

            void init()
            {

                if (verbose_)
                    std::cout << "- Initialization SR-PDE solver" << std::endl;

                // Initialization of the solver
                solver_.setData(df_solver_);
                solver_.init_regularization();
                solver_.init_sampling();
                solver_.init_model();

                return;
            }

            // methods
            void solve()
            {
                if (verbose_)
                    std::cout << "- Solve FR-PDE problem" << std::endl;

                solver_.solve();
                f_ = solver_.f().transpose() / (b_norm_);

                return;
            }

            DMatrix<double> compute(const DMatrix<double> &X, const DVector<double> &b)
            {
                setData(X, b);
                init();
                solve();

                return f_;
            }

            DMatrix<double> compute(const DMatrix<double> &X)
            {
                setData(X);
                init();
                solve();

                return f_;
            }
        };

        template <typename PDE_, typename SamplingDesign_>
        struct model_traits<FRPDE<PDE_, SamplingDesign_>>
        {
            typedef PDE_ PDE;
            typedef SpaceOnly regularization;
            typedef SamplingDesign_ sampling;
            typedef MonolithicSolver solver;
            static constexpr int n_lambda = 1;
        };

        template <typename Model>
        struct is_frpde
        {
            static constexpr bool value = is_instance_of<Model, FRPDE>::value;
        };

    }
}

#endif // __FRPDE_H__
