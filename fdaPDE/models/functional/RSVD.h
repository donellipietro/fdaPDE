#ifndef __RSVD_H__
#define __RSVD_H__

#include <Eigen/SVD>

class RSVD
{
public:
    // Data
    const DMatrix<double> &X_;
    double lambda_;
    const SpMatrix<double> &Psi_;
    const SpMatrix<double> &P_;

    // Intermediate steps matrices
    DMatrix<double> C1_;
    DMatrix<double> D1_;

    // Solutions
    DMatrix<double> H_;
    DMatrix<double> W_;

    // Dimensions
    unsigned int N_;
    unsigned int S_;
    unsigned int K_;

    // Options
    bool verbose_;

    // Constructor
    RSVD(const DMatrix<double> &X, const SpMatrix<double> &Psi, const SpMatrix<double> &P, bool verbose = false)
        : X_(X), Psi_(Psi), P_(P), verbose_(verbose){};

    // Methods

    void init(double lambda)
    {
        // std::cout << "init" << std::endl;

        // Set lambda
        lambda_ = lambda;

        // Dimensions
        N_ = X_.rows();
        S_ = X_.cols();
        K_ = P_.rows();

        // C matrix assembling
        if (verbose_)
            std::cout << "  - C matrix assembling "
                      << "(lambda = " << lambda_ << ")" << std::endl;
        DMatrix<double> C{Psi_.transpose() * Psi_ + lambda_ * P_};

        // C matrix cholesky decomposition
        if (verbose_)
            std::cout << "  - C matrix cholesky decomposition" << std::endl;
        Eigen::LLT<DMatrix<double>> cholesky;
        cholesky.compute(C);
        DMatrix<double> D_{cholesky.matrixL()};

        // D matrix LU decomposition
        if (verbose_)
            std::cout << "  - D matrix LU decomposition" << std::endl;
        Eigen::PartialPivLU<DMatrix<double>> invD;
        invD.compute(D_);

        // D matrix inversion
        if (verbose_)
            std::cout << "  - D matrix inversion" << std::endl;
        D1_ = invD.solve(DMatrix<double>::Identity(K_, K_));

        // std::cout << "init" << std::endl;

        return;
    }

    void solve(unsigned int rank)
    {
        // std::cout << "solve" << std::endl;

        if (verbose_)
            std::cout << "  - Rank is set to " << rank << std::endl;

        H_.resize(N_, rank);
        W_.resize(K_, rank);

        // SVD
        if (verbose_)
            std::cout << "  - SVD" << std::endl;
        Eigen::JacobiSVD<DMatrix<double>> svd(X_ * Psi_ * D1_.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);

        // Results, H
        if (verbose_)
            std::cout << "  - Results, H" << std::endl;
        H_ = svd.matrixU().leftCols(rank);

        // Results, W
        if (verbose_)
            std::cout << "  - Results, W" << std::endl;
        W_ = (svd.singularValues().head(rank).asDiagonal() * svd.matrixV().leftCols(rank).transpose() * D1_).transpose();

        // std::cout << "solve" << std::endl;

        return;
    }

    // Getters
    DMatrix<double> H() const { return H_; }
    DMatrix<double> W() const { return W_; }
};

#endif // __RSVD_H__