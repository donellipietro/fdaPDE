// initialization
template <typename PDE, typename RegularizationType,
          typename SamplingDesign, typename lambda_selection_strategy>
void FPLSR<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>::init_model()
{

    // require zero-mean response and covariates
    this->setCenter(true);

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
        V_.col(h) = pe.s();

        // compute the latent variable
        // std::cout << "latent variable" << std::endl;
        T_.col(h) = E() * Psi() * W_.col(h);
        double tTt = T_.col(h).squaredNorm();

        // regression
        // std::cout << "regression" << std::endl;
        C_.col(h) = this->smoother_.compute(E(), T_.col(h)).transpose();
        D_.col(h) = F().transpose() * T_.col(h) / tTt;

        // deflation
        // std::cout << "deflation" << std::endl;
        E() -= T_.col(h) * C_.col(h).transpose() * PsiTD();
        F() -= T_.col(h) * D_.col(h).transpose();
    }

    auto invAUX = (C_.transpose() * PsiTPsi() * W_).partialPivLu();
    this->B_ = W_ * invAUX.solve(D_.transpose());

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
