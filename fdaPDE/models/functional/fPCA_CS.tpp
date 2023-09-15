template <typename PDE, typename RegularizationType,
          typename SamplingDesign, typename lambda_selection_strategy>
void FPCA_CS<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>::init_model()
{

    if (verbose_)
        std::cout << "\nfPCA (Closed Form Solution)" << std::endl;

    // Dimensions
    N_ = X().rows();
    S_ = X().cols();
    K_ = n_basis();

    // Penalty matrix
    if (mass_lumping_)
    {

        if (verbose_)
            std::cout << "- Penalty matrix assembling (Mass Lumping)" << std::endl;

        // Room for the solution
        SpMatrix<double> invR0;
        invR0.resize(K_, K_);
        invR0.reserve(K_);

        // Triplet list to fill sparse matrix
        std::vector<fdaPDE::Triplet<double>> tripletList;
        tripletList.reserve(K_);
        for (std::size_t i = 0; i < K_; ++i)
            tripletList.emplace_back(i, i, 1 / R0().col(i).sum());

        // Finalize construction
        invR0.setFromTriplets(tripletList.begin(), tripletList.end());
        invR0.makeCompressed();

        // Penalty matrix
        P_ = R1() * invR0 * R1();
    }
    else
    {

        if (verbose_)
            std::cout << "- Penalty matrix assembling (complete)" << std::endl;

        fdaPDE::SparseLU<SpMatrix<double>> invR0;
        invR0.compute(R0());
        P_ = R1() * invR0.solve(R1());
    }

    return;
}

template <typename PDE, typename RegularizationType,
          typename SamplingDesign, typename lambda_selection_strategy>
void FPCA_CS<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>::normalize_results_i(std::size_t i)
{

    double f_n_norm = 0;
    if constexpr (is_space_only<decltype(*this)>::value)
        f_n_norm = std::sqrt(W_.col(i).dot(R0() * W_.col(i)));
    else
        f_n_norm = loadings_.col(i).norm();

    if (coefficients_position_ != 2)
    {
        W_.col(i) /= f_n_norm;
        loadings_.col(i) /= f_n_norm;
    }
    if (coefficients_position_ == 1)
    {
        scores_.col(i) *= f_n_norm;
    }

    coefficients_.insert(i, i) = f_n_norm;

    return;
}

template <typename PDE, typename RegularizationType,
          typename SamplingDesign, typename lambda_selection_strategy>
void FPCA_CS<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>::normalize_results()
{

    for (std::size_t i = 0; i < n_pc_; i++)
        normalize_results_i(i);

    return;
}

// solution in case of fixed \lambda
template <typename PDE, typename RegularizationType,
          typename SamplingDesign, typename lambda_selection_strategy>
void FPCA_CS<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>::solve_(fixed_lambda)
{

    // Resolution
    if (iterative_)
    {
        // Data
        DMatrix<double> X = data().template get<double>(OBSERVATIONS_BLK);

        if (verbose_)
            std::cout << "- RSVD (Iterative)" << std::endl;

        // RSVD initialization
        const unsigned int rank = 1;
        RSVD rsvd(X, Psi(not_nan()), P_, verbose_);
        rsvd.init(lambda()[0]);

        for (std::size_t i = 0; i < n_pc_; i++)
        {

            if (verbose_)
                std::cout << "  Component " << i + 1 << ":" << std::endl;

            // Scores and loadings computation
            rsvd.solve(1);
            W_.col(i) = rsvd.W().col(0);
            loadings_.col(i) = Psi(not_nan()) * W_.col(i);
            scores_.col(i) = rsvd.H().col(0);

            // Deflation
            X -= scores_.col(i) * loadings_.col(i).transpose();

            // Normalization
            normalize_results_i(i);
        }
    }
    else
    {
        // RSVD
        if (verbose_)
            std::cout << "- RSVD (Monolithic)" << std::endl;

        // RSVD initialization
        RSVD rsvd(data().template get<double>(OBSERVATIONS_BLK), Psi(not_nan()), P_, verbose_);
        rsvd.init(lambda()[0]);

        // Scores and loadings computation
        rsvd.solve(n_pc_);
        W_ = rsvd.W();
        loadings_ = Psi(not_nan()) * W_;
        scores_ = rsvd.H();

        // Normalization
        if (verbose_)
            std::cout << "- Scores and Loadings normalization" << std::endl;
        normalize_results();
    }

    return;
}

// best \lambda for PC choosen according to GCV index
template <typename PDE, typename RegularizationType,
          typename SamplingDesign, typename lambda_selection_strategy>
void FPCA_CS<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>::solve_(gcv_lambda_selection)
{
    /*
    ProfilingEstimation<decltype(*this)> pe(*this, tol_, max_iter_);
    BlockFrame<double, int> data_ = data();
    // wrap GCV into a ScalarField accepted by OPT module
    const std::size_t n_lambda = n_smoothing_parameters<RegularizationType>::value;
    ScalarField<n_lambda> f;
    f = [&pe, &data_](const SVector<n_lambda> &p) -> double
    {
        // find vectors s,f minimizing \norm_F{Y - s^T*f}^2 + (s^T*s)*P(f) fixed \lambda = p
        pe.compute(data_, p);
        return pe.gcv(); // return GCV at convergence
    };
    GridOptimizer<n_lambda> opt; // optimization algorithm
    // Principal Components computation
    for (std::size_t i = 0; i < n_pc_; i++)
    {
        opt.optimize(f, lambdas()); // select optimal \lambda for i-th PC
        // compute and store results given estimated optimal \lambda
        pe.compute(data_, opt.optimum());
        loadings_.col(i) = pe.f_n() / pe.f_n_norm();
        scores_.col(i) = pe.s() * pe.f_n_norm();
        // subtract computed PC from data
        data_.get<double>(OBSERVATIONS_BLK) -= scores_.col(i) * loadings_.col(i).transpose();
    }
    return;
    */
}

// best \lambda for PC choosen according to K-fold CV strategy, uses the reconstruction error on test set as CV score
template <typename PDE, typename RegularizationType,
          typename SamplingDesign, typename lambda_selection_strategy>
void FPCA_CS<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>::solve_(kcv_lambda_selection)
{
    /*
    // number of smoothing parameters
    const std::size_t n_lambda = n_smoothing_parameters<RegularizationType>::value;

    if (iterative_)
    {
        // routine executed by the CV-engine to produce the model score
        std::function<double(DVector<double>, BlockFrame<double, int>, BlockFrame<double, int>)> cv_score =
            [this](const DVector<double> &lambda,
                   const BlockFrame<double, int> &train_df,
                   const BlockFrame<double, int> &test_df) -> double
        {
            SVector<n_lambda> p(lambda.data());

            // if (verbose_)
            //    std::cout << "  lambda = " << p[0] << std::endl;

            // References to Training set e Test set
            const DMatrix<double> &X_train = train_df.get<double>(OBSERVATIONS_BLK);
            unsigned int N_train = X_train.rows();
            const DMatrix<double> &X_test = test_df.get<double>(OBSERVATIONS_BLK);
            unsigned int N_test = X_test.rows();

            // Room for solutions training set
            DMatrix<double> W_train;
            DMatrix<double> loadings_train;
            DMatrix<double> scores_train;
            W_train.resize(K_, 1);
            loadings_train.resize(S_, 1);
            scores_train.resize(N_train, 1);

            // fPCA on the training set
            RSVD rsvd(X_train, p[0], 1, Psi(not_nan()), P_, false);
            rsvd.init();
            rsvd.solve();
            W_train = rsvd.W();
            loadings_train = Psi(not_nan()) * W_train;

            // Test set scores vectors
            DMatrix<double> scores_test = X_test * loadings_train;

            // Evaluate reconstruction error on test set
            return (X_test - scores_test * loadings_train.transpose()).squaredNorm() / N_test;
        };

        // Define K-fold algorithm
        KFoldCV cv(10); // allow user-defined number of folds!
        std::vector<DVector<double>> lambdas_;
        lambdas_.reserve(lambdas().size());
        for (const auto &l : lambdas())
            lambdas_.emplace_back(Eigen::Map<const DVector<double>>(l.data(), n_lambda, 1));

        // Principal Components computation
        if (verbose_)
            std::cout << "- lambda selection" << std::endl;
        BlockFrame<double, int> data_ = data();
        for (std::size_t i = 0; i < n_pc_; i++)
        {
            cv.compute(lambdas_, data_, cv_score, false); // select optimal smoothing level

            if (verbose_)
                std::cout << "  Component " << i + 1 << " lambda selected: " << cv.optimum() << std::endl;

            RSVD rsvd(data_.get<double>(OBSERVATIONS_BLK), cv.optimum()[0], 1, Psi(not_nan()), P_, verbose_);
            rsvd.init();
            rsvd.solve();

            W_.col(i) = rsvd.W().col(0);
            loadings_.col(i) = Psi(not_nan()) * W_.col(i);
            scores_.col(i) = rsvd.H().col(0);
            normalize_results_i(i);

            // Subtract computed PC from data
            if (coefficients_position_ == 0)
                data_.get<double>(OBSERVATIONS_BLK) -= coefficients_.coeff(i, i) * scores_.col(i) * loadings_.col(i).transpose();
            else
                data_.get<double>(OBSERVATIONS_BLK) -= scores_.col(i) * loadings_.col(i).transpose();
        }
    }
    else
    {
        // routine executed by the CV-engine to produce the model score
        std::function<double(DVector<double>, BlockFrame<double, int>, BlockFrame<double, int>)> cv_score =
            [this](const DVector<double> &lambda,
                   const BlockFrame<double, int> &train_df,
                   const BlockFrame<double, int> &test_df) -> double
        {
            SVector<n_lambda> p(lambda.data());

            if (verbose_)
                std::cout << "  lambda = " << p[0] << std::endl;

            // References to Training set e Test set
            const DMatrix<double> &X_train = train_df.get<double>(OBSERVATIONS_BLK);
            unsigned int N_train = X_train.rows();
            const DMatrix<double> &X_test = test_df.get<double>(OBSERVATIONS_BLK);
            unsigned int N_test = X_test.rows();

            // Room for solutions training set
            DMatrix<double> W_train;
            DMatrix<double> loadings_train;
            DMatrix<double> scores_train;
            W_train.resize(K_, n_pc_);
            loadings_train.resize(S_, n_pc_);
            scores_train.resize(N_train, n_pc_);

            // fPCA on the training set
            RSVD rsvd(X_train, p[0], n_pc_, Psi(not_nan()), P_, false);
            rsvd.init();
            rsvd.solve();
            W_train = rsvd.W();
            loadings_train = Psi(not_nan()) * W_train;

            // Test set scores vectors
            DMatrix<double> scores_test = X_test * loadings_train;

            // Evaluate reconstruction error on test set
            return (X_test - scores_test * loadings_train.transpose()).squaredNorm() / N_test;
        };

        // define K-fold algorithm
        KFoldCV cv(10); // allow user-defined number of folds!
        std::vector<DVector<double>> lambdas_;
        lambdas_.reserve(lambdas().size());
        for (const auto &l : lambdas())
            lambdas_.emplace_back(Eigen::Map<const DVector<double>>(l.data(), n_lambda, 1));

        // Principal Components computation
        if (verbose_)
            std::cout << "- lambda selection" << std::endl;
        cv.compute(lambdas_, data(), cv_score, false); // select optimal smoothing level

        if (verbose_)
            std::cout << "- lambda selected: " << cv.optimum() << std::endl;

        // RSVD
        if (verbose_)
            std::cout << "- RSVD (Monolithic)" << std::endl;

        RSVD rsvd(data().template get<double>(OBSERVATIONS_BLK), cv.optimum()[0], n_pc_, Psi(not_nan()), P_, verbose_);
        rsvd.init();
        rsvd.solve();
        W_ = rsvd.W();
        loadings_ = Psi(not_nan()) * W_;
        scores_ = rsvd.H();

        // Normalization
        if (verbose_)
            std::cout << "- Scores and Loadings normalization" << std::endl;

        normalize_results();
    }

    return;

    */
}

// finds solution to FPCA_CS problem, dispatch to solver depending on \lambda selection criterion
template <typename PDE, typename RegularizationType,
          typename SamplingDesign, typename lambda_selection_strategy>
void FPCA_CS<PDE, RegularizationType, SamplingDesign, lambda_selection_strategy>::solve()
{
    // pre-allocate space
    W_.resize(K_, n_pc_);
    loadings_.resize(S_, n_pc_);
    scores_.resize(N_, n_pc_);
    coefficients_.resize(n_pc_, n_pc_);
    coefficients_.reserve(n_pc_);

    // dispatch to desired solution strategy
    solve_(lambda_selection_strategy());
    return;
}
