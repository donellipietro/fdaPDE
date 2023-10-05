// Initialization
template <typename PDE, typename RegularizationType,
          typename SamplingDesign, typename lambda_selection_strategy>
void FPLSR<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>::init_model()
{

    if (this->verbose_)
        std::cout << "- Model initialization" << std::endl;

    // Require zero-mean response and covariates
    this->set_center(true);

    // Residuals definition
    df_residuals_.insert<double>(OBSERVATIONS_BLK, this->Y());
    df_residuals_.insert<double>(DESIGN_MATRIX_BLK, this->X());

    // Pre-allocate space
    W_.resize(this->K_, H_);
    V_.resize(this->L_, H_);
    T_.resize(this->N_, H_);
    C_.resize(this->K_, H_);
    D_.resize(this->L_, H_);

    // Spatial matrices
    PsiTPsi_ = PsiTD(not_nan()) * Psi(not_nan());
    invPsiTPsi_.compute(PsiTPsi_);

    return;
}

// Solution in case of fixed \lambda
template <typename PDE, typename RegularizationType,
          typename SamplingDesign, typename lambda_selection_strategy>
void FPLSR<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>::solve_(fixed_lambda)
{

    // Define internal solver
    ProfilingEstimation<decltype(*this)> pe(*this, tol_, max_iter_);

    // Solver's data
    BlockFrame<double, int> M;
    M.insert<double>(OBSERVATIONS_BLK, DMatrix<double>(this->L_, this->S_));

    // wrap GCV into a ScalarField accepted by OPT module
    const std::size_t n_lambda = n_smoothing_parameters<RegularizationType>::value;
    ScalarField<n_lambda> f;
    f = [&pe, &M](const SVector<n_lambda> &p) -> double
    {
        // find vectors s,f minimizing \norm_F{Y - s^T*f}^2 + (s^T*s)*P(f) fixed \lambda = p
        pe.compute(M, p);
        return pe.gcv(); // return GCV at convergence
    };
    GridOptimizer<n_lambda> opt; // optimization algorithm

    if (this->verbose_)
        std::cout << "- Latent components computation" << std::endl;

    // Latent Components computation
    for (std::size_t h = 0; h < H_; h++)
    {

        if (this->verbose_)
            std::cout << "  - Component " << h + 1 << std::endl;

        // Compute directions
        if (this->verbose_)
            std::cout << "    - Directions";
        M.get<double>(OBSERVATIONS_BLK) = F().transpose() * E(); // M = Y^T*X

        SVector<1> best_lambda;
        if (lambdas().size() > 1)
        {
            opt.optimize(f, lambdas()); // select optimal \lambda
            best_lambda = opt.optimum();
        }
        else
        {
            best_lambda = lambdas().front();
        }
        if (this->verbose_)
            std::cout << " (lambda selected = " << best_lambda[0] << ")" << std::endl;
        pe.compute(M, best_lambda);

        W_.col(h) = pe.f() / pe.f_norm(); // / pe.f().norm();
        V_.col(h) = pe.s();

        // Compute the latent variable
        if (this->verbose_)
            std::cout << "    - Latent variable" << std::endl;
        if (this->full_functional_)
            T_.col(h) = E() * R0() * W_.col(h);
        else
            T_.col(h) = E() * Psi(not_nan()) * W_.col(h);
        double tTt = T_.col(h).squaredNorm();

        // Regression
        if (this->verbose_)
            std::cout << "    - Regression" << std::endl;
        D_.col(h) = F().transpose() * T_.col(h) / tTt;
        if (this->full_functional_)
            C_.col(h) = E().transpose() * T_.col(h) / tTt;
        else
        {
            if (smoothing_regression_)
                C_.col(h) = this->smoother_.tune_and_compute(E(), T_.col(h), lambdas_smoothing_regression_).transpose();
            else
                C_.col(h) = invPsiTPsi().solve(PsiTD(not_nan()) * E().transpose() * T_.col(h)) / tTt;
        }

        // Deflation
        if (this->verbose_)
            std::cout << "    - Deflation" << std::endl;
        E() -= T_.col(h) * C_.col(h).transpose() * PsiTD(not_nan());
        F() -= T_.col(h) * D_.col(h).transpose();
    }

    // Functional coefficients matrix computation
    if (this->verbose_)
        std::cout << "- Functional coefficients matrix computation" << std::endl;
    if (this->full_functional_)
    {
        auto invAUX = (C_.transpose() * R0() * W_).partialPivLu();
        this->B_ = W_ * invAUX.solve(D_.transpose());
    }
    else
    {
        auto invAUX = (C_.transpose() * PsiTPsi() * W_).partialPivLu();
        this->B_ = W_ * invAUX.solve(D_.transpose());
    }

    return;
}

// best \lambda for PC choosen according to K-fold CV strategy, uses the reconstruction error on test set as CV score
template <typename PDE, typename RegularizationType,
          typename SamplingDesign, typename lambda_selection_strategy>
void FPLSR<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>::solve_(kcv_lambda_selection)
{

    /*

    // std::cout << "solve FPLSR KCV" << std::endl;

    // define internal solver
    ProfilingEstimation<decltype(*this)> pe(*this, tol_, max_iter_);

    // number of smoothing parameters
    const std::size_t n_lambda = n_smoothing_parameters<RegularizationType>::value;

    // set smoother smoothing parameter
    if (this->smoother_.lambdaS() != lambda_smoothing_regression_)
    {
        // std::cout << "lambda set to " << lambda_smoothing_regression_ << std::endl;
        this->smoother_.setLambdaS(lambda_smoothing_regression_);
    }
    // std::cout << "lambda smoother_regression: " << this->smoother_.lambdaS() << std::endl;
    // std::cout << "lambda regression_required: " << lambda_smoothing_regression_ << std::endl;

    // solver's data
    BlockFrame<double, int> M;
    M.insert<double>(OBSERVATIONS_BLK, DMatrix<double>(this->L_, this->S_));

    // routine executed by the CV-engine to produce the model score
    std::function<double(DVector<double>, BlockFrame<double, int>, BlockFrame<double, int>)> cv_score =
        [&pe, &M, this](const DVector<double> &lambda,
                        const BlockFrame<double, int> &train_df,
                        const BlockFrame<double, int> &test_df) -> double
    {
        // std::cout << "CV-engine" << std::endl;

        std::cout << "|" << std::flush;

        // get references to train and test sets
        const DMatrix<double> &E_train = train_df.get<double>(DESIGN_MATRIX_BLK);
        const DMatrix<double> &F_train = train_df.get<double>(OBSERVATIONS_BLK);
        const DMatrix<double> &E_test = test_df.get<double>(DESIGN_MATRIX_BLK);
        const DMatrix<double> &F_test = test_df.get<double>(OBSERVATIONS_BLK);

        M.get<double>(OBSERVATIONS_BLK) = F_train.transpose() * E_train;

        SVector<n_lambda> p(lambda.data());
        // find vectors s,f minimizing \norm_F{Y - s^T*f}^2 + (s^T*s)*P(f)
        pe.compute(M, p);

        // store result
        // std::cout << "directions" << std::endl;
        DVector<double> w{pe.f() / pe.f_norm()};
        DVector<double> v{pe.s()};

        // compute the latent variable
        // std::cout << "latent variable" << std::endl;
        DVector<double> t_train;
        if (this->full_functional_)
            t_train = E_train * R0() * w;
        else
            t_train = E_train * Psi(not_nan()) * w;

        double tTt = t_train.squaredNorm();

        // regression
        // std::cout << "regression" << std::endl;
        DVector<double> d = F_train.transpose() * t_train / tTt;
        DVector<double> c;
        if (this->full_functional_)
            c = E_train.transpose() * t_train / tTt;
        else
        {
            if (smoothing_regression_)
                c = this->smoother_.tune_and_compute(E_train, t_train).transpose();
            else
                c = invPsiTPsi().solve(PsiTD(not_nan()) * E_train.transpose() * t_train) / tTt;
        }

        // scores vector
        DVector<double> t_test;
        if (this->full_functional_)
            t_test = E_test * R0() * w;
        else
            t_test = E_test * Psi(not_nan()) * w;

        // std::cout << "CV-engine" << std::endl;

        // evaluate reconstruction error on test set
        return (F_test - t_test * d.transpose()).squaredNorm() / F_test.size();
    };

    // define K-fold algorithm
    KFoldCV cv(10); // allow user-defined number of folds!
    std::vector<DVector<double>> lambdas_;
    lambdas_.reserve(lambdas().size());
    for (const auto &l : lambdas())
        lambdas_.emplace_back(Eigen::Map<const DVector<double>>(l.data(), n_lambda, 1));

    // Latent Components computation
    for (std::size_t h = 0; h < H_; h++)
    {
        std::cout << "Component " << h + 1 << std::endl;

        cv.compute(lambdas_, df_residuals_, cv_score, false); // select optimal smoothing level

        std::cout << std::endl;

        // compute directions
        M.get<double>(OBSERVATIONS_BLK) = F().transpose() * E(); // M = Y^T*X
        pe.compute(M, cv.optimum());

        std::cout << "Optimal lambda: " << cv.optimum() << std::endl;

        // store result
        // std::cout << "directions" << std::endl;
        W_.col(h) = pe.f() / pe.f_norm();
        V_.col(h) = pe.s();

        // compute the latent variable
        // std::cout << "latent variable" << std::endl;
        if (this->full_functional_)
            T_.col(h) = E() * R0() * W_.col(h);
        else
            T_.col(h) = E() * Psi(not_nan()) * W_.col(h);

        double tTt = T_.col(h).squaredNorm();

        // regression
        // std::cout << "regression" << std::endl;
        D_.col(h) = F().transpose() * T_.col(h) / tTt;
        if (this->full_functional_)
            C_.col(h) = E().transpose() * T_.col(h) / tTt;
        else
        {
            if (smoothing_regression_)
                C_.col(h) = this->smoother_.compute(E(), T_.col(h)).transpose();
            else
                C_.col(h) = invPsiTPsi().solve(PsiTD(not_nan()) * E().transpose() * T_.col(h)) / tTt;
        }

        // deflation
        // std::cout << "deflation" << std::endl;
        E() -= T_.col(h) * C_.col(h).transpose() * PsiTD(not_nan());
        F() -= T_.col(h) * D_.col(h).transpose();
    }

    if (this->full_functional_)
    {
        auto invAUX = (C_.transpose() * R0() * W_).partialPivLu();
        this->B_ = W_ * invAUX.solve(D_.transpose());
    }
    else
    {
        auto invAUX = (C_.transpose() * PsiTPsi() * W_).partialPivLu();
        this->B_ = W_ * invAUX.solve(D_.transpose());
    }

    // std::cout << "solve FPLSR KCV" << std::endl;

    */

    return;
}

// finds solution to fPLSR problem, dispatch to solver depending on \lambda selection criterion
template <typename PDE, typename RegularizationType,
          typename SamplingDesign, typename lambda_selection_strategy>
void FPLSR<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>::solve()
{
    // dispatch to desired solution strategy
    solve_(lambda_selection_strategy());
}
