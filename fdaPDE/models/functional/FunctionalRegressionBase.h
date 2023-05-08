#ifndef __FUNCTIONAL_REGRESSION_BASE_H__
#define __FUNCTIONAL_REGRESSION_BASE_H__

#include "../../core/utils/Symbols.h"
#include "FunctionalBase.h"
#include "../ModelBase.h"
#include "../SamplingDesign.h"
#include "../ModelTraits.h"
using fdaPDE::models::select_regularization_type;
#include "../SpaceOnlyBase.h"
#include "../SpaceTimeBase.h"
#include "../space_time/SpaceTimeSeparableBase.h"
#include "../space_time/SpaceTimeParabolicBase.h"
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
            using type = FSRPDE<typename model_traits<Model_>::PDE, model_traits<Model_>::sampling>;
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

            // centering
            bool center_;
            DVector<double> Y_mean_;
            DVector<double> X_mean_;
            BlockFrame<double, int> df_data_;
            BlockFrame<double, int> df_centered_;

            // room for problem solution
            DMatrix<double> B_{}; // estimate of the coefficient functions (K_ x L_ matrix)

        public:
            IMPORT_MODEL_SYMBOLS;
            typedef typename model_traits<Model>::PDE PDE; // PDE used for regularization in space
            typedef typename select_regularization_type<Model>::type RegularizationType;
            typedef SamplingDesign<Model, model_traits<Model>::sampling> SamplingBase;
            using RegularizationType::df_;     // BlockFrame for problem's data storage
            using RegularizationType::idx;     // indices of observations
            using RegularizationType::n_basis; // number of basis function over domain D
            using RegularizationType::pde_;    // differential operator L
            using SamplingBase::Psi;           // matrix of spatial basis evaluation at locations p_1 ... p_N

            FunctionalRegressionBase() = default;
            // space-only constructor
            template <typename U = Model, // fake type to enable substitution in SFINAE
                      typename std::enable_if<
                          std::is_same<typename model_traits<U>::RegularizationType, SpaceOnly>::value,
                          int>::type = 0>
            FunctionalRegressionBase(const PDE &pde) : Base(pde), smoother_(pde)
            {
                // std::cout << "initialization FunctionalRegressionBase" << std::endl;

                // smoother initialization
                smoother_.setLambdaS(1e-5);

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
                smoother_.setLambdaS(1e-4);

                // std::cout << "initialization FunctionalRegressionBase" << std::endl;
            }

            // setters
            void setCenter(bool center) { center_ = center; }
            void setDataExtra(BlockFrame<double, int> df_data) { df_data_ = df_data; }

            // getters
            std::size_t q() const { return df_.hasBlock(DESIGN_MATRIX_BLK) ? df_.template get<double>(DESIGN_MATRIX_BLK).cols() : 0; }
            const DMatrix<double> &Y_original() const { return df_data_.template get<double>(OBSERVATIONS_BLK); }      // original responses
            const DMatrix<double> &Y_centered() const { return df_centered_.template get<double>(OBSERVATIONS_BLK); }  // centered responses
            const DVector<double> &Y_mean() const { return Y_mean_; };                                                 // responses mean
            const DMatrix<double> &Y() const { return center_ ? Y_centered() : Y_original(); }                         // responses used in the model
            const DMatrix<double> &X_original() const { return df_data_.template get<double>(DESIGN_MATRIX_BLK); }     // original covariates
            const DMatrix<double> &X_centered() const { return df_centered_.template get<double>(DESIGN_MATRIX_BLK); } // centered covariates
            const DVector<double> &X_mean() const { return X_mean_; };                                                 // covariates mean
            const DMatrix<double> &X() const { return center_ ? X_centered() : X_original(); }                         // covariates used in the model
            const DMatrix<double> &B() const { return B_; };

            // Call this if the internal status of the model must be updated after a change in the data
            // (Called by ModelBase::setData() and executed after initialization of the block frame)
            void update_to_data()
            {

                // add locations to smoother solver
                if constexpr (is_sampling_pointwise_at_locs<SmootherType>::value)
                {
                    const DMatrix<double> locs = df_.template get<double>(SPACE_LOCATIONS_BLK);
                    smoother_.setLocations(locs);
                }

                // centered data
                // std::cout << "update to data" << std::endl;
                Y_mean_ = Y_original().colwise().mean();
                // std::cout << "update to data" << std::endl;
                X_mean_ = smoother_.compute(X_original()).transpose();
                // std::cout << "update to data" << std::endl;
                const DVector<double> X_mean_at_locations = smoother_.fitted().transpose();
                df_centered_.insert<double>(OBSERVATIONS_BLK, Y_original().rowwise() - Y_mean().transpose());
                df_centered_.insert<double>(DESIGN_MATRIX_BLK, X_original().rowwise() - X_mean_at_locations.transpose());

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
                DMatrix<double> Y_hat = X() * Psi() * B_;
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

                DMatrix<double> Y_hat = X_new.transpose() * Psi() * B_;
                if (center_)
                    Y_hat += Y_mean_.transpose();
                return Y_hat;
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
