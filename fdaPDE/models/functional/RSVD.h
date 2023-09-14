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
    DMatrix<double> C1_;
    DMatrix<double> DT1_;

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
    RSVD(const DMatrix<double> &X, const double &lambda, const unsigned int &rank, const SpMatrix<double> &Psi, const SpMatrix<double> &P, bool verbose = false)
        : X_(X), lambda_(lambda), rank_(rank), Psi_(Psi), P_(P), verbose_(verbose)
    {
        N_ = X_.rows();
        S_ = X_.cols();
        K_ = P_.rows();
        H_.resize(N_, rank_);
        W_.resize(K_, rank_);
    };

    // Methods

    void init()
    {

        // std::cout << "init" << std::endl;

        if (verbose_)
            std::cout << "  - C matrix assembling" << std::endl;
        DMatrix<double> C_{Psi_.transpose() * Psi_ + lambda_ * P_};

        if (verbose_)
            std::cout << "  - C matrix cholesky decomposition" << std::endl;
        Eigen::LLT<DMatrix<double>> cholesky;
        cholesky.compute(C_);
        DMatrix<double> D_{cholesky.matrixL()};

        if (verbose_)
            std::cout << "  - C matrix inversion" << std::endl;
        C1_ = cholesky.solve(DMatrix<double>::Identity(K_, K_));

        if (verbose_)
            std::cout << "  - D matrix LU decomposition" << std::endl;
        Eigen::PartialPivLU<DMatrix<double>> invD;
        invD.compute(D_);

        if (verbose_)
            std::cout << "  - D matrix inversion" << std::endl;
        DT1_ = (invD.solve(DMatrix<double>::Identity(K_, K_))).transpose();

        // std::cout << "init" << std::endl;

        return;
    }

    void solve()
    {
        // std::cout << "solve" << std::endl;

        if (verbose_)
            std::cout << "  - SVD" << std::endl;
        Eigen::JacobiSVD<DMatrix<double>> svd(X_ * Psi_ * DT1_, Eigen::ComputeThinU | Eigen::ComputeThinV);

        if (verbose_)
            std::cout << "  - Results, H" << std::endl;
        H_ = svd.matrixU().leftCols(rank_);

        if (verbose_)
            std::cout << "  - Results, W" << std::endl;
        W_ = (H_.transpose() * X_ * Psi_ * C1_).transpose();

        // std::cout << "solve" << std::endl;

        return;
    }

    // Getters
    DMatrix<double> H() const { return H_; }
    DMatrix<double> W() const { return W_; }
};

#endif // __RSVD_H__