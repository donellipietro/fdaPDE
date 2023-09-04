#ifndef __RSVD_H__
#define __RSVD_H__

#include <Eigen/SVD>

class RSVD
{
public:
    // Data
    const DMatrix<double> &X_;
    const double &lambda_;
    const SpMatrix<double> &Psi_;
    const SpMatrix<double> &P_;
    const unsigned int &rank_;

    // Intermediate steps matrices
    DMatrix<double> C_;
    DMatrix<double> D_;

    // Solutions
    DMatrix<double> H_;
    DMatrix<double> W_;

    // Options
    bool verbose_;

    // Constructor
    RSVD(const DMatrix<double> &X, const double &lambda, const unsigned int &rank, const SpMatrix<double> &Psi, const SpMatrix<double> &P, bool verbose = false)
        : X_(X), lambda_(lambda), rank_(rank), Psi_(Psi), P_(P), verbose_(verbose)
    {
        H_.resize(X_.rows(), rank_);
        W_.resize(X_.cols(), rank_);
    };

    // Methods
    void solve()
    {
        // std::cout << "solve" << std::endl;

        std::size_t K = P_.rows();

        if (verbose_)
            std::cout << "  - C matrix assembling" << std::endl;
        C_ = Psi_.transpose() * Psi_ + lambda_ * P_;

        if (verbose_)
            std::cout << "  - C matrix cholesky decomposition" << std::endl;
        Eigen::LLT<DMatrix<double>> cholesky(C_);
        D_ = cholesky.matrixL();

        if (verbose_)
            std::cout << "  - D matrix LU decomposition" << std::endl;
        Eigen::PartialPivLU<DMatrix<double>> invD;
        invD.compute(D_);

        if (verbose_)
            std::cout << "  - SVD" << std::endl;
        Eigen::JacobiSVD<DMatrix<double>> svd(X_ * Psi_ * (invD.solve(DMatrix<double>::Identity(K, K))).transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);

        if (verbose_)
            std::cout << "  - Results, H" << std::endl;
        H_ = svd.matrixU().leftCols(rank_);

        if (verbose_)
            std::cout << "  - Results, W" << std::endl;
        W_ = cholesky.solve(DMatrix<double>::Identity(K, K)) * Psi_.transpose() * X_.transpose() * H_;

        // std::cout << "solve" << std::endl;
    }

    // Getters
    DMatrix<double> H() const { return H_; }
    DMatrix<double> W() const { return W_; }
};

#endif // __RSVD_H__