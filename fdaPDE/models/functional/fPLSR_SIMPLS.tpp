// initialization
template <typename PDE, typename RegularizationType,
          typename SamplingDesign, typename lambda_selection_strategy>
void FPLSR_SIMPLS<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>::init_model()
{

    // std::cout << "init_model" << std::endl;

    // require zero-mean response and covariates
    this->set_center(true);

    // pre-allocate space
    R_.resize(this->K_, H_);
    Q_.resize(this->L_, H_);
    T_.resize(this->N_, H_);
    U_.resize(this->N_, H_);
    P_.resize(this->K_, H_);
    V_.resize(this->K_, H_);
    h_.resize(this->N_);
    varX_explained_.resize(H_);
    varY_explained_.resize(H_);

    // spatial matrices
    // std::cout << "invPsiTPsi_" << std::endl;
    PsiTPsi_ = PsiTD(not_nan()) * Psi(not_nan());
    invPsiTPsi_.compute(PsiTPsi_);

    if (smoothing_regression_)
    {
        // mass lumping
        // std::cout << "mass lumping" << std::endl;
        SpMatrix<double> R0_lumped;
        R0_lumped.resize(this->K_, this->K_);
        R0_lumped.reserve(this->K_);
        std::vector<fdaPDE::Triplet<double>> tripletList;
        tripletList.reserve(this->K_);
        for (std::size_t i = 0; i < this->K_; ++i)
        {
            tripletList.emplace_back(i, i, R0().row(i).sum());
        }
        R0_lumped.setFromTriplets(tripletList.begin(), tripletList.end());
        R0_lumped.makeCompressed();
        // std::cout << R0().block(0, 0, 10, 10) << std::endl;
        // std::cout << R0_lumped.block(0, 0, 10, 10) << std::endl;
        invR0_.compute(R0_lumped); // R0()

        // R term
        // std::cout << "R_term_" << std::endl;
        R_term_ = R1() * invR0_.solve(R1());

        // energy matrix
        // std::cout << "A_" << std::endl;
        A_ = PsiTPsi_ + lambda_smoothing_regression_ * R_term_;
    }
    else
    {
        A_ = PsiTPsi_;
    }

    // std::cout << "init_model" << std::endl;

    return;
}

// solution in case of fixed \lambda
template <typename PDE, typename RegularizationType,
          typename SamplingDesign, typename lambda_selection_strategy>
void FPLSR_SIMPLS<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>::solve_(fixed_lambda)
{
    // std::cout << "solve FPLSR_SIMPLS" << std::endl;

    // define internal solver
    ProfilingEstimation<decltype(*this)> pe(*this, tol_, max_iter_);

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
    M.insert<double>(OBSERVATIONS_BLK, this->Y().transpose() * this->X_original());

    // Latent Components computation
    for (std::size_t h = 0; h < H_; h++)
    {
        // std::cout << "Component " << h + 1 << ", lambda = " << lambda() << std::endl;

        // compute directions
        pe.compute(M, lambda());

        // store result
        // std::cout << "directions" << std::endl;
        R_.col(h) = pe.f();
        Q_.col(h) = pe.s();

        // compute the x-latent score
        // std::cout << "x-latent score" << std::endl;
        T_.col(h) = this->X_original() * Psi(not_nan()) * R_.col(h);
        T_.col(h) = T_.col(h).array() - T_.col(h).mean();
        const double normT = T_.col(h).norm();
        T_.col(h) = T_.col(h) / normT;

        // normalize x-direction
        R_.col(h) = R_.col(h) / normT;

        // regression
        // std::cout << "regression" << std::endl;
        Q_.col(h) = this->Y().transpose() * T_.col(h);
        if (smoothing_regression_)
            P_.col(h) = this->smoother_.compute(this->X_original(), T_.col(h)).transpose();
        else
            P_.col(h) = invPsiTPsi().solve(PsiTD(not_nan()) * this->X_original().transpose() * T_.col(h));

        // compute the y-latent score
        // std::cout << "y-latent scores" << std::endl;
        U_.col(h) = this->Y() * Q_.col(h);

        // Orthonormal basis for P column space
        // std::cout << "orthonormal basis" << std::endl;
        V_.col(h) = P_.col(h);
        if (h > 0)
        {
            DMatrix<double> Vh = V_.block(0, 0, this->K_, h);     // V_(h-1)
            DMatrix<double> Th = T_.block(0, 0, this->N_, h + 1); // T_(h)
            V_.col(h) -= Vh * (Vh.transpose() * (A_ * V_.col(h)));
            U_.col(h) -= Th * (Th.transpose() * U_.col(h));
        }
        V_.col(h) /= std::sqrt(V_.col(h).transpose() * A_ * V_.col(h));

        // deflation
        // std::cout << "deflation" << std::endl;
        const DVector<double> v_s = Psi(not_nan()) * V_.col(h);
        M.get<double>(OBSERVATIONS_BLK) -= M.get<double>(OBSERVATIONS_BLK) * v_s * v_s.transpose();
    }

    // regression coefficient matrix
    // std::cout << "regression coefficient matrix" << std::endl;
    this->B_ = R_ * Q_.transpose();

    // analysis
    // std::cout << "analysis" << std::endl;
    h_ = (T_ * T_.transpose()).diagonal().array() + 1 / this->N_;
    varX_ = (this->X().transpose() * this->X()).trace();
    varY_ = (this->Y().transpose() * this->Y()).trace();
    varX_explained_ = (Psi(not_nan()) * P_ * P_.transpose() * PsiTD(not_nan())).diagonal() / (this->N_ - 1);
    varY_explained_ = (Q_ * Q_.transpose()).diagonal() / (this->N_ - 1);

    // residuals
    // std::cout << "residuals" << std::endl;
    df_residuals_.insert<double>(OBSERVATIONS_BLK, this->X_original() - reconstructed_field() * PsiTD(not_nan()));
    df_residuals_.insert<double>(DESIGN_MATRIX_BLK, this->Y_original() - this->fitted());

    // std::cout << "solve FPLSR_SIMPLS" << std::endl;

    return;
}

// best \lambda for PC choosen according to GCV index
template <typename PDE, typename RegularizationType,
          typename SamplingDesign, typename lambda_selection_strategy>
void FPLSR_SIMPLS<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>::solve_(gcv_lambda_selection)
{
    // std::cout << "solve FPLSR_SIMPLS" << std::endl;

    // define internal solver
    ProfilingEstimation<decltype(*this)> pe(*this, tol_, max_iter_);

    // define covariance matrix
    BlockFrame<double, int> M;

    // define components index
    std::size_t h = 0;

    // set smoother smoothing parameter
    if (this->smoother_.lambdaS() != lambda_smoothing_regression_)
    {
        // std::cout << "lambda set to " << lambda_smoothing_regression_ << std::endl;
        this->smoother_.setLambdaS(lambda_smoothing_regression_);
    }
    // std::cout << "lambda smoother_regression: " << this->smoother_.lambdaS() << std::endl;
    // std::cout << "lambda regression_required: " << lambda_smoothing_regression_ << std::endl;

    // wrap GCV into a ScalarField accepted by OPT module
    const std::size_t n_lambda = n_smoothing_parameters<RegularizationType>::value;
    ScalarField<n_lambda> f;
    f = [&pe, &M, &h, this](const SVector<n_lambda> &p) -> double
    {
        // find vectors s,f minimizing \norm_F{Y - s^T*f}^2 + (s^T*s)*P(f) fixed \lambda = p
        pe.compute(M, p);

        // store result
        // std::cout << "directions" << std::endl;
        R_.col(h) = pe.f();
        Q_.col(h) = pe.s();

        // compute the x-latent score
        // std::cout << "x-latent score" << std::endl;
        T_.col(h) = this->X_original() * Psi(not_nan()) * R_.col(h);
        T_.col(h) = T_.col(h).array() - T_.col(h).mean();
        const double normT = T_.col(h).norm();
        T_.col(h) = T_.col(h) / normT;

        DMatrix<double> Th = T_.block(0, 0, this->N_, h + 1); // V_(h)

        DMatrix<double> S = Th * Th.transpose();
        double MSE = (this->Y() - S * this->Y()).squaredNorm();

        return this->N_ / (this->N_ - (h + S.trace())) * MSE; // return GCV at convergence
    };

    // optimization algorithm
    GridOptimizer<n_lambda> opt;

    // solver's data
    M.insert<double>(OBSERVATIONS_BLK, this->Y().transpose() * this->X_original());

    // Latent Components computation
    for (h = 0; h < H_; h++)
    {
        std::cout << "Component " << h + 1 << std::endl;

        // select optimal \lambda for h-th latent component
        opt.optimize(f, lambdas());
        std::cout << "lambda_omptimal = " << opt.optimum() << std::endl;

        // compute directions using the optimal \lambda fot h-th latent component
        pe.compute(M, opt.optimum());

        // store result
        // std::cout << "directions" << std::endl;
        R_.col(h) = pe.f();
        Q_.col(h) = pe.s();

        // compute the x-latent score
        // std::cout << "x-latent score" << std::endl;
        T_.col(h) = this->X_original() * Psi(not_nan()) * R_.col(h);
        T_.col(h) = T_.col(h).array() - T_.col(h).mean();
        const double normT = T_.col(h).norm();
        T_.col(h) = T_.col(h) / normT;

        // normalize x-direction
        R_.col(h) = R_.col(h) / normT;

        // regression
        // std::cout << "regression" << std::endl;
        Q_.col(h) = this->Y().transpose() * T_.col(h);
        if (smoothing_regression_)
            P_.col(h) = this->smoother_.compute(this->X_original(), T_.col(h)).transpose();
        else
            P_.col(h) = invPsiTPsi().solve(PsiTD(not_nan()) * this->X_original().transpose() * T_.col(h));

        // compute the y-latent score
        // std::cout << "y-latent scores" << std::endl;
        U_.col(h) = this->Y() * Q_.col(h);

        // Orthonormal basis for P column space
        // std::cout << "orthonormal basis" << std::endl;
        V_.col(h) = P_.col(h);
        if (h > 0)
        {
            DMatrix<double> Vh = V_.block(0, 0, this->K_, h);     // V_(h-1)
            DMatrix<double> Th = T_.block(0, 0, this->N_, h + 1); // T_(h)
            V_.col(h) -= Vh * Vh.transpose() * A_ * V_.col(h);
            U_.col(h) -= Th * Th.transpose() * U_.col(h);
        }
        V_.col(h) /= std::sqrt(V_.col(h).transpose() * A_ * V_.col(h));

        // deflation
        // std::cout << "deflation" << std::endl;
        const DVector<double> v_s = Psi(not_nan()) * V_.col(h);
        M.get<double>(OBSERVATIONS_BLK) -= M.get<double>(OBSERVATIONS_BLK) * v_s * v_s.transpose();
    }

    // regression coefficient matrix
    // std::cout << "regression coefficient matrix" << std::endl;
    this->B_ = R_ * Q_.transpose();

    // analysis
    // std::cout << "analysis" << std::endl;
    h_ = (T_ * T_.transpose()).diagonal().array() + 1 / this->N_;
    varX_ = (this->X().transpose() * this->X()).trace();
    varY_ = (this->Y().transpose() * this->Y()).trace();
    varX_explained_ = (Psi(not_nan()) * P_ * P_.transpose() * PsiTD(not_nan())).diagonal() / (this->N_ - 1);
    varY_explained_ = (Q_ * Q_.transpose()).diagonal() / (this->N_ - 1);

    // residuals
    // std::cout << "residuals" << std::endl;
    df_residuals_.insert<double>(OBSERVATIONS_BLK, this->X_original() - reconstructed_field() * PsiTD(not_nan()));
    df_residuals_.insert<double>(DESIGN_MATRIX_BLK, this->Y_original() - this->fitted());

    // std::cout << "solve FPLSR_SIMPLS" << std::endl;

    return;
}

// finds solution to fPLSR problem, dispatch to solver depending on \lambda selection criterion
template <typename PDE, typename RegularizationType,
          typename SamplingDesign, typename lambda_selection_strategy>
void FPLSR_SIMPLS<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>::solve()
{
    // dispatch to desired solution strategy
    solve_(lambda_selection_strategy());
}
