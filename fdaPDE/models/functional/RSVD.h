#ifndef __RSVD_H__
#define __RSVD_H__

#include <Eigen/SVD>

class RSVD
{
public:
    // Data
    DMatrix<double> X_;
    double lambda_;
    SpMatrix<double> Psi_;
    SpMatrix<double> P_;
    unsigned int rank_;

    // Intermediate steps matrices
    DMatrix<double> C_;
    DMatrix<double> D_;

    // Solutions
    DMatrix<double> W_;
    DMatrix<double> H_;

    // Constructors
    // RSVD(DMatrix<double> X, double lambda, unsigned int rank)
    //     : X_(X), lambda_(lambda), rank_(rank)
    // {
    //     Psi_ = DMatrix<double>::Identity(X_.cols(), X_.cols());
    //     P_ = DMatrix<double>::Zero(X_.cols(), X_.cols());
    // };
    // RSVD(DMatrix<double> X, double lambda, unsigned int rank, SpMatrix<double> P)
    //     : X_(X), lambda_(lambda), rank_(rank), P_(P)
    // {
    //     if (X_.cols() == P_.rows())
    //     {
    //         Psi_ = DMatrix<double>::Identity(X_.cols(), X_.cols());
    //     }
    //     else
    //     {
    //         throw std::runtime_error("Error: Dimensions are wrong.");
    //     }
    // };
    RSVD(DMatrix<double> X, double lambda, unsigned int rank, SpMatrix<double> Psi, SpMatrix<double> P)
        : X_(X), lambda_(lambda), rank_(rank), Psi_(Psi), P_(P){};

    // Methods
    void solve()
    {
        // std::cout << "solve" << std::endl;

        // std::cout << "C" << std::endl;
        C_ = Psi_.transpose() * Psi_ + lambda_ * P_;

        // std::cout << "Cholesky" << std::endl;
        Eigen::LLT<DMatrix<double>> cholesky(C_);
        D_ = cholesky.matrixL();

        Eigen::PartialPivLU<DMatrix<double>> invD_;
        invD_.compute(D_);

        // std::cout << "SVD" << std::endl;
        Eigen::JacobiSVD<DMatrix<double>> svd(X_ * Psi_ * invD_.solve(DMatrix<double>::Identity(P_.rows(), P_.cols())), Eigen::ComputeThinU | Eigen::ComputeThinV);

        // std::cout << "Results" << std::endl;
        H_ = svd.matrixU().leftCols(rank_);
        W_ = H_.transpose() * X_ * Psi_ * cholesky.solve(DMatrix<double>::Identity(P_.rows(), P_.cols()));

        // std::cout << "solve" << std::endl;
    }

    // Setters
    void set_data(DMatrix<double> &X) { X_ = X; }
    void set_lambda(double lambda) { lambda_ = lambda; }
    void set_rank(unsigned int rank) { rank_ = rank; }
    void set_P(SpMatrix<double> &P) { P_ = P; }

    // Getters
    DMatrix<double> scores() const { return H_; }
    DMatrix<double> loadings() const { return W_; }
};

#endif // __RSVD_H__