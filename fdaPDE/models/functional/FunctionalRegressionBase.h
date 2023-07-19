#ifndef __FUNCTIONAL_REGRESSION_BASE_H__
#define __FUNCTIONAL_REGRESSION_BASE_H__

#include "../../core/utils/Symbols.h"
#include "../ModelBase.h"
#include "../SamplingDesign.h"
#include "../ModelTraits.h"
using fdaPDE::models::select_regularization_type;
#include "FunctionalBase.h"
#include "fSRPDE.h"
using fdaPDE::models::FSRPDE;

namespace fdaPDE
{
    namespace models
    {

        template <typename Model>
        struct FSRPDE_smoother
        {
            typedef typename std::decay<Model>::type Model_;
            using type = FSRPDE<typename model_traits<Model_>::PDE, typename model_traits<Model_>::sampling>;
        };

        // base class for any *functional regression* fdaPDE model
        template <typename Model>
        class FunctionalRegressionBase : public FunctionalBase<Model>
        {
        protected:
            typedef FunctionalBase<Model> Base;

            // problem dimensions
            std::size_t L_; // number of responses
            std::size_t S_; // number of observation points
            std::size_t N_; // number of samples
            std::size_t K_; // number of basis

            // smoother
            typedef typename FSRPDE_smoother<Model>::type SmootherType;
            SmootherType smoother_;

            // version implemented in R
            bool full_functional_ = false;

            // smoothing
            bool smoothing_initialization_ = true;
            double lambda_smoothing_initialization_ = 1e-12;

            // centering
            bool center_;
            DVector<double> Y_mean_;
            DVector<double> X_mean_;
            BlockFrame<double, int> df_data_;
            BlockFrame<double, int> df_centered_;

            // room for problem solution
            DMatrix<double> B_; // estimate of the coefficient functions (K_ x L_ matrix)

        public:
            IMPORT_MODEL_SYMBOLS;
            typedef typename model_traits<Model>::PDE PDE; // PDE used for regularization in space
            typedef typename select_regularization_type<Model>::type RegularizationType;
            typedef SamplingDesign<Model, typename model_traits<Model>::sampling> SamplingBase;
            using FunctionalBase<Model>::n_stat_units;
            using RegularizationType::df_;     // BlockFrame for problem's data storage
            using RegularizationType::idx;     // indices of observations
            using RegularizationType::n_basis; // number of basis function over domain D
            using RegularizationType::pde_;    // differential operator L
            using SamplingBase::locs;          // matrix of spatial locations
            using SamplingBase::Psi;           // matrix of spatial basis evaluation at locations p_1 ... p_N
            using SamplingBase::PsiTD;         // block \Psi^T*D

            FunctionalRegressionBase() = default;
            // space-only constructor
            template <typename U = Model, // fake type to enable substitution in SFINAE
                      typename std::enable_if<
                          std::is_same<typename model_traits<U>::regularization, SpaceOnly>::value,
                          int>::type = 0>
            FunctionalRegressionBase(const PDE &pde) : Base(pde), smoother_(pde)
            {
                // std::cout << "initialization FunctionalRegressionBase" << std::endl;

                // smoother initialization
                smoother_.setLambdaS(lambda_smoothing_initialization_);

                // std::cout << "initialization FunctionalRegressionBase" << std::endl;
            };

            // space-time constructor
            /*
            template <typename U = Model, // fake type to enable substitution in SFINAE
                      typename std::enable_if<!std::is_same<typename model_traits<U>::RegularizationType, SpaceOnly>::value,
                                              int>::type = 0>
            FunctionalRegressionBase(const PDE &pde, const DVector<double> &time)
                : select_regularization_type<Model>::type(pde, time),
                  SamplingDesign<Model, model_traits<Model>::sampling>(){};
            */

            // copy constructor, copy only pde object (as a consequence also the problem domain)
            FunctionalRegressionBase(const FunctionalRegressionBase &rhs)
            {
                // std::cout << "initialization FunctionalRegressionBase" << std::endl;

                pde_ = rhs.pde_;
                // smoother initialization
                smoother_.setPDE(pde_);
                smoother_.setLambdaS(lambda_smoothing_initialization_);

                // std::cout << "initialization FunctionalRegressionBase" << std::endl;
            }

            // setters
            void set_center(bool center) { center_ = center; }
            void set_smoothing(bool smoothing) { smoothing_initialization_ = smoothing; }
            void set_smoothing(bool smoothing, double lambda_smoothing_initialization)
            {
                smoothing_initialization_ = smoothing;
                if (smoother_.lambdaS() != lambda_smoothing_initialization)
                {
                    lambda_smoothing_initialization_ = lambda_smoothing_initialization;
                    smoother_.setLambdaS(lambda_smoothing_initialization_);
                }
            }

            // getters
            std::size_t q() const { return df_.hasBlock(DESIGN_MATRIX_BLK) ? df_.template get<double>(DESIGN_MATRIX_BLK).cols() : 0; }
            const DMatrix<double> &Y_original() const { return df_.template get<double>(OBSERVATIONS_BLK); }           // original responses
            const DMatrix<double> &Y_centered() const { return df_centered_.template get<double>(OBSERVATIONS_BLK); }  // centered responses
            const DVector<double> &Y_mean() const { return Y_mean_; };                                                 // responses mean
            const DMatrix<double> &Y() const { return center_ ? Y_centered() : Y_original(); }                         // responses used in the model
            const DMatrix<double> &X_original() const { return df_.template get<double>(DESIGN_MATRIX_BLK); }          // original covariates
            const DMatrix<double> &X_centered() const { return df_centered_.template get<double>(DESIGN_MATRIX_BLK); } // centered covariates
            const DVector<double> &X_mean() const { return X_mean_; };                                                 // covariates mean
            const DMatrix<double> &X() const { return center_ ? X_centered() : X_original(); }                         // covariates used in the model
            const DMatrix<double> &B() const { return B_; };

            // Call this if the internal status of the model must be updated after a change in the data
            // (Called by ModelBase::setData() and executed after initialization of the block frame)
            void init_data()
            {

                // add locations to smoother solver
                if constexpr (is_sampling_pointwise_at_locs<SmootherType>::value)
                {
                    // std::cout << "set spatial locations" << std::endl;
                    const DMatrix<double> locs = this->locs();
                    smoother_.set_spatial_locations(locs);
                    // std::cout << "set spatial locations" << std::endl;
                }

                // centered data
                // std::cout << "update to data" << std::endl;
                Y_mean_ = Y_original().colwise().mean();
                df_centered_.insert<double>(OBSERVATIONS_BLK, Y_original().rowwise() - Y_mean().transpose());
                // std::cout << "update to data" << std::endl;
                if (smoothing_initialization_)
                {
                    // std::cout << "lambda smoother: " << smoother_.lambdaS() << std::endl;
                    // std::cout << "lambda required: " << lambda_smoothing_initialization_ << std::endl;
                    X_mean_ = smoother_.compute(X_original()).transpose();
                    const DVector<double> X_mean_at_locations = smoother_.fitted().transpose();
                    df_centered_.insert<double>(DESIGN_MATRIX_BLK, X_original().rowwise() - X_mean_at_locations.transpose());
                }
                else
                {
                    X_mean_ = X_original().colwise().mean();
                    df_centered_.insert<double>(DESIGN_MATRIX_BLK, X_original().rowwise() - X_mean_.transpose());
                }
                // std::cout << "update to data" << std::endl;

                // dimensions
                L_ = Y().cols();
                S_ = X().cols();
                N_ = Y().rows();
                K_ = n_basis();

                // solution
                B_.resize(K_, L_);
            }

            // computes fitted values
            // \hat y = y_mean = X*Psi*B + y_mean
            DMatrix<double> fitted() const
            {
                DMatrix<double> Y_hat(N_, L_);
                if (full_functional_)
                    Y_hat = X() * R0() * B_;
                else
                    Y_hat = X() * Psi(not_nan()) * B_;

                if (center_)
                    Y_hat = Y_hat.rowwise() + Y_mean_.transpose();
                return Y_hat;
            }

            // compute prediction of model for a new unseen function:
            // \hat y_{n+1} = x_{n+1}^T*Psi*B + y_mean
            DMatrix<double> predict(const DVector<double> &covs)
            {
                DVector<double> X_new{covs};
                if (center_)
                    X_new -= X_mean();

                DMatrix<double> Y_hat(1, L_);
                if (full_functional_)
                    Y_hat = X_new.transpose() * R0() * B_;
                else
                    Y_hat = X_new.transpose() * Psi(not_nan()) * B_;

                if (center_)
                    Y_hat += Y_mean_.transpose();
                return Y_hat;
            }

            // functional models' missing data logic
            void init_nan()
            {
                /*
                this->nan_idxs_.clear(); // clean previous missingness structure
                this->nan_idxs_.resize(n_stat_units());
                this->PsiNaN_.resize(n_stat_units());
                // \Psi matrix dimensions
                std::size_t n = Psi(not_nan()).rows();
                std::size_t N = Psi(not_nan()).cols();
                // for i-th statistical unit, analyze missingness structure and set \Psi_i
                for (std::size_t i = 0; i < n_stat_units(); ++i)
                {
                    // derive missingness pattern for i-th statistical unit
                    for (std::size_t j = 0; j < n_locs(); ++j)
                    {
                        if (std::isnan(X_original()(i, j))) // requires -ffast-math compiler flag to be disabled
                            this->nan_idxs_[i].insert(j);
                    }

                    // NaN detected for this unit, start assembly
                    if (!this->nan_idxs_[i].empty())
                    {
                        for (std::size_t i = 0; i < n_stat_units(); ++i)
                        {
                            this->PsiNaN_[i].resize(n, N); // reserve space
                            std::vector<fdaPDE::Triplet<double>> tripletList;
                            tripletList.reserve(n * N);
                            for (int k = 0; k < Psi(not_nan()).outerSize(); ++k)
                                for (SpMatrix<double>::InnerIterator it(Psi(not_nan()), k); it; ++it)
                                {
                                    if (this->nan_idxs_[i].find(it.row()) == this->nan_idxs_[i].end())
                                        // no missing data at this location for i-th statistical unit
                                        tripletList.emplace_back(it.row(), it.col(), it.value());
                                }
                            // finalize construction
                            this->PsiNaN_[i].setFromTriplets(tripletList.begin(), tripletList.end());
                            this->PsiNaN_[i].makeCompressed();
                        }
                    }
                    // otherwise no matrix is assembled, full \Psi is returned by Psi(std::size_t) getter
                }
                */
                return;
            }
        };

        // trait to detect if a type is a functional regression model
        template <typename T>
        struct is_functional_regression_model
        {
            static constexpr bool value = fdaPDE::is_base_of_template<FunctionalRegressionBase, T>::value;
        };

    }
}

#endif // __FUNCTIONAL_REGRESSION_BASE_H__
