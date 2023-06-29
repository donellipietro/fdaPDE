#include <cstddef>
#include <gtest/gtest.h> // testing framework
#include <unsupported/Eigen/SparseExtra>

#include "../fdaPDE/core/utils/Symbols.h"
#include "../fdaPDE/core/utils/IO/CSVReader.h"
#include "../fdaPDE/core/FEM/PDE.h"
using fdaPDE::core::FEM::PDE;
#include "core/MESH/Mesh.h"
#include "../fdaPDE/models/functional/fPLSR_SIMPLS.h"
using fdaPDE::models::FPLSR_SIMPLS;
#include "../fdaPDE/models/SamplingDesign.h"
#include "../../fdaPDE/models/ModelTraits.h"

#include "../utils/MeshLoader.h"
using fdaPDE::testing::MeshLoader;
#include "../utils/Constants.h"
using fdaPDE::testing::DOUBLE_TOLERANCE;
#include "../utils/Utils.h"
using fdaPDE::testing::almost_equal;

#include <cmath>
#include <random>
#include <fstream>
#include <filesystem>

namespace Test_FPLSR_SIMPLS_base
{

    // Normalize sign of the columns
    DMatrix<double> ns(DMatrix<double> M)
    {
        for (std::size_t j = 0; j < M.cols(); ++j)
        {
            double sign = std::abs(M.col(j)[0]) / M.col(j)[0];
            M.col(j) *= sign;
        }
        return M;
    }

    void compare(std::string &test_directory,
                 bool VERBOSE,
                 bool ONLY_FINAL_RESULTS,
                 DMatrix<double> &R,
                 DMatrix<double> &Q,
                 DMatrix<double> &T,
                 DMatrix<double> &U,
                 DMatrix<double> &P,
                 DMatrix<double> &V,
                 DMatrix<double> &Y_hat,
                 DMatrix<double> &Y_mean,
                 DMatrix<double> &X_hat,
                 DMatrix<double> &X_mean,
                 DMatrix<double> &B_hat,
                 std::string type = "")
    {

        CSVFile<double> file;
        CSVReader<double> reader{};

        file = reader.parseFile(test_directory + "Y_hat" + type + ".csv");
        DMatrix<double> expected_Y_hat = file.toEigen();
        if (VERBOSE)
        {
            std::cout << std::endl;
            std::cout << "||||||| Y ||||||||:" << std::endl;
            std::cout << "Expected:" << std::endl;
            std::cout << expected_Y_hat.topRows(5) << std::endl;
            std::cout << "Obtained version:" << std::endl;
            std::cout << Y_hat.topRows(5) << std::endl;
            std::cout << "Error norm: " << (expected_Y_hat - Y_hat).lpNorm<Eigen::Infinity>() << std::endl;
            std::cout << "----------------" << std::endl;
            std::cout << std::endl;
        }
        EXPECT_TRUE(almost_equal(expected_Y_hat, Y_hat));

        file = reader.parseFile(test_directory + "Y_mean" + type + ".csv");
        DMatrix<double> expected_Y_mean = file.toEigen();
        if (VERBOSE)
        {
            std::cout << std::endl;
            std::cout << "||||||| Y_mean ||||||||:" << std::endl;
            std::cout << "Expected:" << std::endl;
            std::cout << expected_Y_mean.topRows(1) << std::endl;
            std::cout << "Obtained:" << std::endl;
            std::cout << Y_mean.topRows(1) << std::endl;
            std::cout << "Error norm: " << (expected_Y_mean - Y_mean).lpNorm<Eigen::Infinity>() << std::endl;
            std::cout << "----------------" << std::endl;
            std::cout << std::endl;
        }
        EXPECT_TRUE(almost_equal(expected_Y_mean, Y_mean));

        file = reader.parseFile(test_directory + "X_hat" + type + ".csv");
        DMatrix<double> expected_X_hat = file.toEigen();
        if (VERBOSE)
        {
            std::cout << std::endl;
            std::cout << "||||||| X ||||||||:" << std::endl;
            std::cout << "Expected:" << std::endl;
            std::cout << expected_X_hat.block(0, 0, 5, 5) << std::endl;
            std::cout << "Obtained:" << std::endl;
            std::cout << X_hat.block(0, 0, 5, 5) << std::endl;
            std::cout << "Error norm: " << (expected_X_hat - X_hat).lpNorm<Eigen::Infinity>() << std::endl;
            std::cout << "----------------" << std::endl;
            std::cout << std::endl;
        }
        EXPECT_TRUE(almost_equal(expected_X_hat, X_hat));

        file = reader.parseFile(test_directory + "X_mean" + type + ".csv");
        DMatrix<double> expected_X_mean = file.toEigen().transpose();
        if (VERBOSE)
        {
            std::cout << std::endl;
            std::cout << "||||||| X_mean ||||||||:" << std::endl;
            std::cout << "Expected:" << std::endl;
            std::cout << expected_X_mean.block(0, 0, 1, 5) << std::endl;
            std::cout << "Obtained:" << std::endl;
            std::cout << X_mean.block(0, 0, 1, 5) << std::endl;
            std::cout << "Error norm: " << (expected_X_mean - X_mean).lpNorm<Eigen::Infinity>() << std::endl;
            std::cout << "----------------" << std::endl;
            std::cout << std::endl;
        }
        EXPECT_TRUE(almost_equal(expected_X_mean, X_mean));

        file = reader.parseFile(test_directory + "B_hat" + type + ".csv");
        DMatrix<double> expected_B_hat = file.toEigen();
        if (VERBOSE)
        {
            std::cout << std::endl;
            std::cout << "||||||| B ||||||||:" << std::endl;
            std::cout << "Expected:" << std::endl;
            std::cout << expected_B_hat.topRows(5) << std::endl;
            std::cout << "Obtained:" << std::endl;
            std::cout << B_hat.topRows(5) << std::endl;
            std::cout << "Error norm: " << (expected_B_hat - B_hat).lpNorm<Eigen::Infinity>() << std::endl;
            std::cout << "----------------" << std::endl;
            std::cout << std::endl;
        }
        EXPECT_TRUE(almost_equal(expected_B_hat, B_hat));

        if (!ONLY_FINAL_RESULTS)
        {

            file = reader.parseFile(test_directory + "R" + type + ".csv");
            DMatrix<double> expected_R = file.toEigen();
            if (VERBOSE)
            {
                std::cout << std::endl;
                std::cout << "||||||| R ||||||||:" << std::endl;
                std::cout << "Expected:" << std::endl;
                std::cout << expected_R.topRows(5) << std::endl;
                std::cout << "Obtained:" << std::endl;
                std::cout << R.topRows(5) << std::endl;
                std::cout << "Error norm: " << (Test_FPLSR_SIMPLS_base::ns(expected_R) - Test_FPLSR_SIMPLS_base::ns(R)).lpNorm<Eigen::Infinity>() << std::endl;
                std::cout << "----------------" << std::endl;
                std::cout << std::endl;
            }
            EXPECT_TRUE(almost_equal(ns(expected_R), ns(R)));

            file = reader.parseFile(test_directory + "Q" + type + ".csv");
            DMatrix<double> expected_Q = file.toEigen();
            if (VERBOSE)
            {
                std::cout << std::endl;
                std::cout << "||||||| Q ||||||||:" << std::endl;
                std::cout << "Expected:" << std::endl;
                std::cout << expected_Q.topRows(1) << std::endl;
                std::cout << "Obtained:" << std::endl;
                std::cout << Q.topRows(1) << std::endl;
                std::cout << "Error norm: " << (Test_FPLSR_SIMPLS_base::ns(expected_Q) - Test_FPLSR_SIMPLS_base::ns(Q)).lpNorm<Eigen::Infinity>() << std::endl;
                std::cout << "----------------" << std::endl;
                std::cout << std::endl;
            }
            EXPECT_TRUE(almost_equal(ns(expected_Q), ns(Q)));

            file = reader.parseFile(test_directory + "T" + type + ".csv");
            DMatrix<double> expected_T = file.toEigen();
            if (VERBOSE)
            {
                std::cout << std::endl;
                std::cout << "||||||| T ||||||||:" << std::endl;
                std::cout << "Expected:" << std::endl;
                std::cout << expected_T.topRows(5) << std::endl;
                std::cout << "Obtained:" << std::endl;
                std::cout << T.topRows(5) << std::endl;
                std::cout << "Error norm: " << (Test_FPLSR_SIMPLS_base::ns(expected_T) - Test_FPLSR_SIMPLS_base::ns(T)).lpNorm<Eigen::Infinity>() << std::endl;
                std::cout << "----------------" << std::endl;
                std::cout << std::endl;
            }
            EXPECT_TRUE(almost_equal(ns(expected_T), ns(T)));

            file = reader.parseFile(test_directory + "U" + type + ".csv");
            DMatrix<double> expected_U = file.toEigen();
            if (VERBOSE)
            {
                std::cout << std::endl;
                std::cout << "||||||| U ||||||||:" << std::endl;
                std::cout << "Expected:" << std::endl;
                std::cout << expected_U.topRows(5) << std::endl;
                std::cout << "Obtained:" << std::endl;
                std::cout << U.topRows(5) << std::endl;
                std::cout << "Error norm: " << (Test_FPLSR_SIMPLS_base::ns(expected_U) - Test_FPLSR_SIMPLS_base::ns(U)).lpNorm<Eigen::Infinity>() << std::endl;
                std::cout << "----------------" << std::endl;
                std::cout << std::endl;
            }
            EXPECT_TRUE(almost_equal(ns(expected_U), ns(U)));

            file = reader.parseFile(test_directory + "P" + type + ".csv");
            DMatrix<double> expected_P = file.toEigen();
            if (VERBOSE)
            {
                std::cout << std::endl;
                std::cout << "||||||| P ||||||||:" << std::endl;
                std::cout << "Expected:" << std::endl;
                std::cout << expected_P.topRows(5) << std::endl;
                std::cout << "Obtained:" << std::endl;
                std::cout << P.topRows(5) << std::endl;
                std::cout << "Error norm: " << (Test_FPLSR_SIMPLS_base::ns(expected_P) - Test_FPLSR_SIMPLS_base::ns(P)).lpNorm<Eigen::Infinity>() << std::endl;
                std::cout << "----------------" << std::endl;
                std::cout << std::endl;
            }
            EXPECT_TRUE(almost_equal(ns(expected_P), ns(P)));

            file = reader.parseFile(test_directory + "V" + type + ".csv");
            DMatrix<double> expected_V = file.toEigen();
            if (VERBOSE)
            {
                std::cout << std::endl;
                std::cout << "||||||| V ||||||||:" << std::endl;
                std::cout << "Expected:" << std::endl;
                std::cout << expected_V.topRows(1) << std::endl;
                std::cout << "Obtained:" << std::endl;
                std::cout << V.topRows(1) << std::endl;
                std::cout << "Error norm: " << (Test_FPLSR_SIMPLS_base::ns(expected_V) - Test_FPLSR_SIMPLS_base::ns(V)).lpNorm<Eigen::Infinity>() << std::endl;
                std::cout << "----------------" << std::endl;
                std::cout << std::endl;
            }
            EXPECT_TRUE(almost_equal(ns(expected_V), ns(V)));
        }
    }

}

/* test 0:
   comparison with multivariate results (SIMPLS)
   domain:       unit square [0,1] x [0,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   BC:           no
   order FE:     1
 */
TEST(FPLSR_SIMPLS, Test0_Laplacian_GeostatisticalAtNodes_comparison_with_multivariate_SIMPLS)
{
    // define domain and regularizing PDE
    MeshLoader<Mesh2D<>> domain("unit_square");
    auto L = Laplacian();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements() * 3, 1);
    PDE problem(domain.mesh, L, u); // definition of regularizing PDE

    // define statistical model
    FPLSR_SIMPLS<decltype(problem),
                 SpaceOnly,
                 fdaPDE::models::GeoStatMeshNodes,
                 fdaPDE::models::fixed_lambda>
        model(problem);

    // Tests
    std::string tests_directory = "data/models/FPLSR_SIMPLS/2D_test0/";
    bool VERBOSE = false;
    std::vector<unsigned int> tests{1, 2, 3, 4, 5, 6};

    // Multivariate model
    double lambda = 1e-15;
    bool full_funtional = false;
    bool smoothing = false;

    for (unsigned int i : tests)
    {

        if (VERBOSE)
        {
            std::cout << "##########" << std::endl;
            std::cout << "# Test " << i << " #" << std::endl;
            std::cout << "##########" << std::endl;
        }

        std::string test_directory = tests_directory + "test" + std::to_string(i) + "/";

        // load data from .csv files
        CSVReader<double> reader{};
        CSVFile<double> yFile;
        yFile = reader.parseFile(test_directory + "Y.csv");
        DMatrix<double> Y = yFile.toEigen();
        CSVFile<double> xFile;
        xFile = reader.parseFile(test_directory + "X.csv");
        DMatrix<double> X = xFile.toEigen();

        // set smoothing parameter
        model.setLambdaS(lambda);

        // disable smoothing for initialization and regression
        model.set_smoothing(smoothing);

        // full_functional: true -> harold's implementation, false -> correct implementation
        model.set_full_functional(full_funtional);

        // set model data
        BlockFrame<double, int> df_data;
        df_data.insert(OBSERVATIONS_BLK, Y);
        df_data.insert(DESIGN_MATRIX_BLK, X);
        model.setData(df_data);

        // solve smoothing problem
        model.init();
        model.solve();

        //   **  test correctness of computed results  **   //

        // Results
        DMatrix<double> R{model.R()};
        DMatrix<double> Q{model.Q()};
        DMatrix<double> T{model.T()};
        DMatrix<double> U{model.U()};
        DMatrix<double> P{model.P()};
        DMatrix<double> V{model.V()};
        DMatrix<double> Y_hat{model.fitted()};
        DMatrix<double> Y_mean{model.Y_mean().transpose()};
        DMatrix<double> X_hat{model.reconstructed_field()};
        DMatrix<double> X_mean{model.X_mean().transpose()};
        DMatrix<double> B_hat{model.B()};

        Test_FPLSR_SIMPLS_base::compare(test_directory, VERBOSE, false, R, Q, T, U, P, V, Y_hat, Y_mean, X_hat, X_mean, B_hat, "_multivariate_SIMPLS");
    }
}