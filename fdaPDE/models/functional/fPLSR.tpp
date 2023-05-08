// initialization
template <typename PDE, typename RegularizationType,
          Sampling SamplingDesign, typename lambda_selection_strategy>
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
    PsiTPsi_ = this->PsiTD() * this->Psi();
    invPsiTPsi_.compute(PsiTPsi_);
    return;
}

// solution in case of fixed \lambda
template <typename PDE, typename RegularizationType,
          Sampling SamplingDesign, typename lambda_selection_strategy>
void FPLSR<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>::solve_(fixed_lambda)
{
    // std::cout << "solve FPLSR" << std::endl;

    // define internal solver
    FPIREM<decltype(*this)> solver(*this);

    solver.setTolerance(tol_);
    solver.setMaxIterations(max_iter_);

    // solver's data
    BlockFrame<double, int> M;
    M.insert<double>(OBSERVATIONS_BLK, DMatrix<double>::Zero(this->S_, this->L_)); // Covariace matrix
    if constexpr (SamplingDesign == GeoStatLocations)
        M.insert<double>(SPACE_LOCATIONS_BLK, this->df_.template get<double>(SPACE_LOCATIONS_BLK));

    // Latent Components computation
    for (std::size_t h = 0; h < H_; h++)
    {
        // std::cout << "Component " << h + 1 << ", lambda = " << lambda() << std::endl;

        // compute directions
        M.get<double>(OBSERVATIONS_BLK) = this->E().transpose() * this->F(); // M^T = (Y^T*X)^T = X^T*Y
        solver.setData(M);
        solver.setLambda(lambda());
        solver.solve();

        // store result
        // std::cout << "directions" << std::endl;
        W_.col(h) = solver.f();
        V_.col(h) = solver.scores();

        // compute the latent variable
        // std::cout << "latent variable" << std::endl;
        T_.col(h) = this->E() * this->Psi() * W_.col(h);
        double tTt = T_.col(h).squaredNorm();

        // regression
        // std::cout << "regression" << std::endl;
        C_.col(h) = this->smoother_.compute(E(), T_.col(h)).transpose();
        D_.col(h) = this->F().transpose() * T_.col(h) / tTt;

        // deflation
        // std::cout << "deflation" << std::endl;
        this->E() -= T_.col(h) * C_.col(h).transpose() * this->PsiTD();
        this->F() -= T_.col(h) * D_.col(h).transpose();
    }

    auto invAUX = (C_.transpose() * PsiTPsi() * W_).partialPivLu();
    this->B_ = W_ * invAUX.solve(D_.transpose());

    // std::cout << "solve FPLSR" << std::endl;

    return;
}

// best \lambda for PC choosen according to GCV index
template <typename PDE, typename RegularizationType,
          Sampling SamplingDesign, typename lambda_selection_strategy>
void FPLSR<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>::solve_(gcv_lambda_selection)
{
    /*
    // define internal solver
    typedef typename FPIREM<decltype(*this)>::SmootherType SmootherType; // solver used to smooth loadings
    FPIREM<decltype(*this)> solver(*this);
    solver.setTolerance(tol_);
    solver.setMaxIterations(max_iter_);

    BlockFrame<double, int> df = data();
    // df.insert<double>(OBSERVATIONS_BLK, y().transpose()); // move original data to a n_subject() x n_obs() matrix
    //  define GCV objective for internal smoothing model used by FPIREM
    GCV<SmootherType, StochasticEDF<SmootherType>> GCV(solver.smoother(), 100);
    std::cout << ":)" << std::endl;
    // wrap GCV into a ScalarField accepted by the OPT module
    ScalarField<model_traits<SmootherType>::n_lambda> f;
    f = [&GCV, &solver](const SVector<model_traits<SmootherType>::n_lambda> &p) -> double
    {
        // set \lambda and solve smoothing problem on solver
        std::cout << "lambda: " << p << std::endl;
        solver.setLambda(p);
        std::cout << "settato" << std::endl;
        solver.solve();
        std::cout << "risolto" << std::endl;
        std::cout << "GCV" << std::endl;
        std::cout << GCV.eval() << std::endl;
        // return evaluation of GCV at point
        return GCV.eval();
    };

    // define GCV optimization algorithm
    GridOptimizer<model_traits<SmootherType>::n_lambda> opt;
    // Principal Components computation
    for (std::size_t i = 0; i < n_pc_; i++)
    {
        solver.setData(df);
        opt.optimize(f, lambda_vect_); // select optimal lambda for this PC
        // compute result given estimated optimal \lambda
        std::cout << "lambda found \n"
                  << opt.optimum() << std::endl;

        solver.setLambda(opt.optimum());
        solver.solve(); // find minimum of \norm_F{Y - s^T*f}^2 + (s^T*s)*P(f)
        // store result
        loadings_.col(i) = solver.loadings();
        scores_.col(i) = solver.scores();
        // subtract computed PC from data
        df.get<double>(OBSERVATIONS_BLK) -= loadings_.col(i) * scores_.col(i).transpose();
    }
    */
    return;
}

// best \lambda for PC choosen according to K fold CV error
template <typename PDE, typename RegularizationType,
          Sampling SamplingDesign, typename lambda_selection_strategy>
void FPLSR<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>::solve_(kcv_lambda_selection)
{
    /*
    // define internal solver
    FPIREM<decltype(*this)> solver(*this);
    solver.setTolerance(tol_);
    solver.setMaxIterations(max_iter_);

    BlockFrame<double, int> df;
    df.insert<double>(OBSERVATIONS_BLK, y().transpose()); // n_subject() x n_obs() matrix
    KFoldCV<decltype(solver)> lambda_selector(solver, 5);

    for (std::size_t i = 0; i < n_pc_; i++)
    {
        solver.setData(df);
        SVector<1> optimal_lambda = lambda_selector.compute(lambda_vect_, PCScoreCV());
        // compute result given estimated optimal \lambda
        solver.setLambda(optimal_lambda);
        solver.solve(); // find minimum of \norm_F{Y - s^T*f}^2 + (s^T*s)*P(f)
        // store result
        loadings_.col(i) = solver.loadings();
        scores_.col(i) = solver.scores();
        // subtract computed PC from data
        df.get<double>(OBSERVATIONS_BLK) -= scores_.col(i) * loadings_.col(i).transpose();
    }

    std::cout << scores_ << std::endl;
    */
    return;
}

// finds solution to fPCA problem, dispatch to solver depending on \lambda selection criterion
template <typename PDE, typename RegularizationType,
          Sampling SamplingDesign, typename lambda_selection_strategy>
void FPLSR<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>::solve()
{
    // dispatch to desired solution strategy
    solve_(lambda_selection_strategy());
}
