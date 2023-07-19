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

    // Constructor
    RSVD(const DMatrix<double> &X, const double &lambda, const unsigned int &rank, const SpMatrix<double> &Psi, const SpMatrix<double> &P)
        : X_(X), lambda_(lambda), rank_(rank), Psi_(Psi), P_(P)
    {
        H_.resize(X_.rows(), rank_);
        W_.resize(rank_, X_.cols());
    };

    // Methods
    void solve()
    {
        // std::cout << "solve" << std::endl;
        std::size_t K = P_.rows();

        // std::cout << "C" << std::endl;
        C_ = Psi_.transpose() * Psi_ + lambda_ * P_;
        // std::cout << C_.block(0,0,5,5) << std::endl;
        // std::cout << std::endl;

        // std::cout << "Cholesky" << std::endl;
        Eigen::LLT<DMatrix<double>> cholesky(C_);
        D_ = cholesky.matrixL();
        // std::cout << D_.block(0,0,5,5) << std::endl;
        // std::cout << std::endl;

        Eigen::PartialPivLU<DMatrix<double>> invD;
        invD.compute(D_);

        // std::cout << "SVD" << std::endl;
        Eigen::JacobiSVD<DMatrix<double>> svd(X_ * Psi_ * (invD.solve(DMatrix<double>::Identity(K, K))).transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);

        // std::cout << "Results, H" << std::endl;
        H_ = svd.matrixU().leftCols(rank_);
        // std::cout << H_.topRows(5) << std::endl;
        // std::cout << std::endl;

        // std::cout << "Results, W" << std::endl;
        W_ = H_.transpose() * X_ * Psi_ * cholesky.solve(DMatrix<double>::Identity(P_.rows(), P_.cols()));
        // std::cout << W_.leftCols(5) << std::endl;
        // std::cout << std::endl;

        // std::cout << "solve" << std::endl;
    }

    // Getters
    DMatrix<double> scores() const { return H_; }
    DMatrix<double> loadings() const { return W_; }
};

#endif // __RSVD_H__