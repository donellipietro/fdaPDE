// initialization
template <typename PDE, typename RegularizationType,
          typename SamplingDesign, typename lambda_selection_strategy>
void FPLSR<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>::init_model()
{

    // std::cout << "init_model" << std::endl;

    // require zero-mean response and covariates
    this->set_center(true);

    // residuals definition
    df_residuals_.insert<double>(OBSERVATIONS_BLK, this->Y());
    df_residuals_.insert<double>(DESIGN_MATRIX_BLK, this->X());

    // pre-allocate space
    W_.resize(this->K_, H_);
    V_.resize(this->L_, H_);
    T_.resize(this->N_, H_);
    C_.resize(this->K_, H_);
    D_.resize(this->L_, H_);

    // spatial matrices
    PsiTPsi_ = PsiTD() * Psi();
    invPsiTPsi_.compute(PsiTPsi_);

    // std::cout << "init_model" << std::endl;

    return;
}

// solution in case of fixed \lambda
template <typename PDE, typename RegularizationType,
          typename SamplingDesign, typename lambda_selection_strategy>
void FPLSR<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>::solve_(fixed_lambda)
{
    // std::cout << "solve FPLSR" << std::endl;

    // define internal solver
    ProfilingEstimation<decltype(*this)> pe(*this, tol_, max_iter_);

    // solver's data
    DMatrix<double> M(this->L_, this->S_);

    // Latent Components computation
    for (std::size_t h = 0; h < H_; h++)
    {
        // std::cout << "Component " << h + 1 << ", lambda = " << lambda() << std::endl;

        // compute directions
        M = F().transpose() * E(); // M = Y^T*X
        pe.compute(M, lambda());

        // store result
        // std::cout << "directions" << std::endl;
        W_.col(h) = pe.f();
        V_.col(h) = pe.s() / pe.s().norm();

        // compute the latent variable
        // std::cout << "latent variable" << std::endl;
        if (this->full_functional_)
            T_.col(h) = E() * R0() * W_.col(h);
        else
            T_.col(h) = E() * Psi() * W_.col(h);

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
                C_.col(h) = invPsiTPsi().solve(PsiTD() * E().transpose() * T_.col(h)) / tTt;
        }

        // deflation
        // std::cout << "deflation" << std::endl;
        E() -= T_.col(h) * C_.col(h).transpose() * PsiTD();
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

    // std::cout << "solve FPLSR" << std::endl;

    return;
}

// best \lambda for PC choosen according to GCV index
template <typename PDE, typename RegularizationType,
          typename SamplingDesign, typename lambda_selection_strategy>
void FPLSR<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>::solve_(gcv_lambda_selection)
{
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
