#include <cstddef>
#include <gtest/gtest.h> // testing framework
#include <unsupported/Eigen/SparseExtra>

#include "../fdaPDE/core/utils/Symbols.h"
#include "../fdaPDE/core/utils/IO/CSVReader.h"
#include "../fdaPDE/core/FEM/PDE.h"
using fdaPDE::core::FEM::PDE;
#include "core/MESH/Mesh.h"
#include "../fdaPDE/models/functional/fPLSR.h"
using fdaPDE::models::FPLSR;
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

namespace Test_fPLSR
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

    void compare_NIPALS(std::string &test_directory,
                        std::string &method_name,
                        bool VERBOSE,
                        bool ONLY_FINAL_RESULTS,
                        DMatrix<double> &W,
                        DMatrix<double> &V,
                        DMatrix<double> &T,
                        DMatrix<double> &C,
                        DMatrix<double> &D,
                        DMatrix<double> &Y_hat1,
                        DMatrix<double> &Y_hat2,
                        DMatrix<double> &Y_mean,
                        DMatrix<double> &X_hat,
                        DMatrix<double> &X_mean,
                        DMatrix<double> &B_hat,
                        std::string type = "")
    {

        CSVFile<double> file;
        CSVReader<double> reader{};

        file = reader.parseFile(test_directory + method_name + "/" + "Y_hat" + type + ".csv");
        DMatrix<double> expected_Y_hat = file.toEigen();
        if (VERBOSE)
        {
            std::cout << std::endl;
            std::cout << "||||||| Y ||||||||:" << std::endl;
            std::cout << "Expected:" << std::endl;
            std::cout << expected_Y_hat.topRows(5) << std::endl;
            std::cout << "Obtained version 1:" << std::endl;
            std::cout << Y_hat1.topRows(5) << std::endl;
            std::cout << "Obtained version 2:" << std::endl;
            std::cout << Y_hat2.topRows(5) << std::endl;
            std::cout << "Error norm 1: " << (expected_Y_hat - Y_hat1).lpNorm<Eigen::Infinity>() << std::endl;
            std::cout << "Error norm 2: " << (expected_Y_hat - Y_hat2).lpNorm<Eigen::Infinity>() << std::endl;
            std::cout << "----------------" << std::endl;
            std::cout << std::endl;
        }
        EXPECT_TRUE(almost_equal(expected_Y_hat, Y_hat1));
        EXPECT_TRUE(almost_equal(expected_Y_hat, Y_hat2));

        file = reader.parseFile(test_directory + method_name + "/" + "Y_mean" + type + ".csv");
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

        file = reader.parseFile(test_directory + method_name + "/" + "X_hat" + type + ".csv");
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

        file = reader.parseFile(test_directory + method_name + "/" + "X_mean" + type + ".csv");
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

        file = reader.parseFile(test_directory + method_name + "/" + "B_hat" + type + ".csv");
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

            file = reader.parseFile(test_directory + method_name + "/" + "W" + type + ".csv");
            DMatrix<double> expected_W = file.toEigen();
            if (VERBOSE)
            {
                std::cout << std::endl;
                std::cout << "||||||| W ||||||||:" << std::endl;
                std::cout << "Expected:" << std::endl;
                std::cout << expected_W.topRows(5) << std::endl;
                std::cout << "Obtained:" << std::endl;
                std::cout << W.topRows(5) << std::endl;
                std::cout << "Error norm: " << (Test_fPLSR::ns(expected_W) - Test_fPLSR::ns(W)).lpNorm<Eigen::Infinity>() << std::endl;
                std::cout << "----------------" << std::endl;
                std::cout << std::endl;
            }
            EXPECT_TRUE(almost_equal(ns(expected_W), ns(W)));

            file = reader.parseFile(test_directory + method_name + "/" + "V" + type + ".csv");
            DMatrix<double> expected_V = file.toEigen();
            if (VERBOSE)
            {
                std::cout << std::endl;
                std::cout << "||||||| V ||||||||:" << std::endl;
                std::cout << "Expected:" << std::endl;
                std::cout << expected_V.topRows(1) << std::endl;
                std::cout << "Obtained:" << std::endl;
                std::cout << V.topRows(1) << std::endl;
                std::cout << "Error norm: " << (Test_fPLSR::ns(expected_V) - Test_fPLSR::ns(V)).lpNorm<Eigen::Infinity>() << std::endl;
                std::cout << "----------------" << std::endl;
                std::cout << std::endl;
            }
            EXPECT_TRUE(almost_equal(ns(expected_V), ns(V)));

            file = reader.parseFile(test_directory + method_name + "/" + "T" + type + ".csv");
            DMatrix<double> expected_T = file.toEigen();
            if (VERBOSE)
            {
                std::cout << std::endl;
                std::cout << "||||||| T ||||||||:" << std::endl;
                std::cout << "Expected:" << std::endl;
                std::cout << expected_T.topRows(5) << std::endl;
                std::cout << "Obtained:" << std::endl;
                std::cout << T.topRows(5) << std::endl;
                std::cout << "Error norm: " << (Test_fPLSR::ns(expected_T) - Test_fPLSR::ns(T)).lpNorm<Eigen::Infinity>() << std::endl;
                std::cout << "----------------" << std::endl;
                std::cout << std::endl;
            }
            // EXPECT_TRUE(almost_equal(ns(expected_T), ns(T)));

            file = reader.parseFile(test_directory + method_name + "/" + "C" + type + ".csv");
            DMatrix<double> expected_C = file.toEigen();
            if (VERBOSE)
            {
                std::cout << std::endl;
                std::cout << "||||||| C ||||||||:" << std::endl;
                std::cout << "Expected:" << std::endl;
                std::cout << expected_C.topRows(5) << std::endl;
                std::cout << "Obtained:" << std::endl;
                std::cout << C.topRows(5) << std::endl;
                std::cout << "Error norm: " << (Test_fPLSR::ns(expected_C) - Test_fPLSR::ns(C)).lpNorm<Eigen::Infinity>() << std::endl;
                std::cout << "----------------" << std::endl;
                std::cout << std::endl;
            }
            // EXPECT_TRUE(almost_equal(ns(expected_C), ns(C)));

            file = reader.parseFile(test_directory + method_name + "/" + "D" + type + ".csv");
            DMatrix<double> expected_D = file.toEigen();
            if (VERBOSE)
            {
                std::cout << std::endl;
                std::cout << "||||||| D ||||||||:" << std::endl;
                std::cout << "Expected:" << std::endl;
                std::cout << expected_D.topRows(1) << std::endl;
                std::cout << "Obtained:" << std::endl;
                std::cout << D.topRows(1) << std::endl;
                std::cout << "Error norm: " << (Test_fPLSR::ns(expected_D) - Test_fPLSR::ns(D)).lpNorm<Eigen::Infinity>() << std::endl;
                std::cout << "----------------" << std::endl;
                std::cout << std::endl;
            }
            // EXPECT_TRUE(almost_equal(ns(expected_D), ns(D)));
        }
    }

    void compare_SIMPLS(std::string &test_directory,
                        std::string &method_name,
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

        file = reader.parseFile(test_directory + method_name + "/" + "Y_hat" + type + ".csv");
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

        file = reader.parseFile(test_directory + method_name + "/" + "Y_mean" + type + ".csv");
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

        file = reader.parseFile(test_directory + method_name + "/" + "X_hat" + type + ".csv");
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

        file = reader.parseFile(test_directory + method_name + "/" + "X_mean" + type + ".csv");
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

        file = reader.parseFile(test_directory + method_name + "/" + "B_hat" + type + ".csv");
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

            file = reader.parseFile(test_directory + method_name + "/" + "R" + type + ".csv");
            DMatrix<double> expected_R = file.toEigen();
            if (VERBOSE)
            {
                std::cout << std::endl;
                std::cout << "||||||| R ||||||||:" << std::endl;
                std::cout << "Expected:" << std::endl;
                std::cout << expected_R.topRows(5) << std::endl;
                std::cout << "Obtained:" << std::endl;
                std::cout << R.topRows(5) << std::endl;
                std::cout << "Error norm: " << (Test_fPLSR::ns(expected_R) - Test_fPLSR::ns(R)).lpNorm<Eigen::Infinity>() << std::endl;
                std::cout << "----------------" << std::endl;
                std::cout << std::endl;
            }
            EXPECT_TRUE(almost_equal(ns(expected_R), ns(R)));

            file = reader.parseFile(test_directory + method_name + "/" + "Q" + type + ".csv");
            DMatrix<double> expected_Q = file.toEigen();
            if (VERBOSE)
            {
                std::cout << std::endl;
                std::cout << "||||||| Q ||||||||:" << std::endl;
                std::cout << "Expected:" << std::endl;
                std::cout << expected_Q.topRows(1) << std::endl;
                std::cout << "Obtained:" << std::endl;
                std::cout << Q.topRows(1) << std::endl;
                std::cout << "Error norm: " << (Test_fPLSR::ns(expected_Q) - Test_fPLSR::ns(Q)).lpNorm<Eigen::Infinity>() << std::endl;
                std::cout << "----------------" << std::endl;
                std::cout << std::endl;
            }
            EXPECT_TRUE(almost_equal(ns(expected_Q), ns(Q)));

            file = reader.parseFile(test_directory + method_name + "/" + "T" + type + ".csv");
            DMatrix<double> expected_T = file.toEigen();
            if (VERBOSE)
            {
                std::cout << std::endl;
                std::cout << "||||||| T ||||||||:" << std::endl;
                std::cout << "Expected:" << std::endl;
                std::cout << expected_T.topRows(5) << std::endl;
                std::cout << "Obtained:" << std::endl;
                std::cout << T.topRows(5) << std::endl;
                std::cout << "Error norm: " << (Test_fPLSR::ns(expected_T) - Test_fPLSR::ns(T)).lpNorm<Eigen::Infinity>() << std::endl;
                std::cout << "----------------" << std::endl;
                std::cout << std::endl;
            }
            EXPECT_TRUE(almost_equal(ns(expected_T), ns(T)));

            file = reader.parseFile(test_directory + method_name + "/" + "U" + type + ".csv");
            DMatrix<double> expected_U = file.toEigen();
            if (VERBOSE)
            {
                std::cout << std::endl;
                std::cout << "||||||| U ||||||||:" << std::endl;
                std::cout << "Expected:" << std::endl;
                std::cout << expected_U.topRows(5) << std::endl;
                std::cout << "Obtained:" << std::endl;
                std::cout << U.topRows(5) << std::endl;
                std::cout << "Error norm: " << (Test_fPLSR::ns(expected_U) - Test_fPLSR::ns(U)).lpNorm<Eigen::Infinity>() << std::endl;
                std::cout << "----------------" << std::endl;
                std::cout << std::endl;
            }
            EXPECT_TRUE(almost_equal(ns(expected_U), ns(U)));

            file = reader.parseFile(test_directory + method_name + "/" + "P" + type + ".csv");
            DMatrix<double> expected_P = file.toEigen();
            if (VERBOSE)
            {
                std::cout << std::endl;
                std::cout << "||||||| P ||||||||:" << std::endl;
                std::cout << "Expected:" << std::endl;
                std::cout << expected_P.topRows(5) << std::endl;
                std::cout << "Obtained:" << std::endl;
                std::cout << P.topRows(5) << std::endl;
                std::cout << "Error norm: " << (Test_fPLSR::ns(expected_P) - Test_fPLSR::ns(P)).lpNorm<Eigen::Infinity>() << std::endl;
                std::cout << "----------------" << std::endl;
                std::cout << std::endl;
            }
            EXPECT_TRUE(almost_equal(ns(expected_P), ns(P)));

            file = reader.parseFile(test_directory + method_name + "/" + "V" + type + ".csv");
            DMatrix<double> expected_V = file.toEigen();
            if (VERBOSE)
            {
                std::cout << std::endl;
                std::cout << "||||||| V ||||||||:" << std::endl;
                std::cout << "Expected:" << std::endl;
                std::cout << expected_V.topRows(1) << std::endl;
                std::cout << "Obtained:" << std::endl;
                std::cout << V.topRows(1) << std::endl;
                std::cout << "Error norm: " << (Test_fPLSR::ns(expected_V) - Test_fPLSR::ns(V)).lpNorm<Eigen::Infinity>() << std::endl;
                std::cout << "----------------" << std::endl;
                std::cout << std::endl;
            }
            EXPECT_TRUE(almost_equal(ns(expected_V), ns(V)));
        }
    }

}

/* test 1:
   Comparison R implementation vs C++ implementation
   domain:       unit square [0,1] x [0,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   BC:           no
   order FE:     1
 */
TEST(FPLSR, Test1_RvsCpp)
{
    // define domain and regularizing PDE
    MeshLoader<Mesh2D<>> domain("unit_square");
    auto L = Laplacian();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements() * 3, 1);
    PDE problem(domain.mesh, L, u);

    // define statistical model
    FPLSR<decltype(problem),
          SpaceOnly,
          fdaPDE::models::GeoStatMeshNodes,
          fdaPDE::models::fixed_lambda>
        model(problem);

    // tests
    std::string tests_directory = "data/models/FPLSR/2D_test/";
    std::string method_name = "r1fpls";
    bool VERBOSE = false;
    std::vector<unsigned int> tests{1, 2, 3, 4, 5};

    // R functional model
    std::vector<SVector<1>> lambdas{SVector<1>{10.0}};
    bool smoothing = false;
    bool full_functional = true;

    for (unsigned int i : tests)
    {

        if (VERBOSE)
        {
            std::cout << "##########" << std::endl;
            std::cout << "# Test " << i << " #" << std::endl;
            std::cout << "##########" << std::endl;
        }

        std::string test_directory = tests_directory + "test_" + std::to_string(i) + "/";

        // load data from .csv files
        CSVReader<double> reader{};
        CSVFile<double> yFile;
        yFile = reader.parseFile(test_directory + "Y.csv");
        DMatrix<double> Y = yFile.toEigen();
        CSVFile<double> xFile;
        xFile = reader.parseFile(test_directory + "X.csv");
        DMatrix<double> X = xFile.toEigen();

        // set model options
        model.set_verbose(VERBOSE);
        model.setLambda(lambdas);
        model.set_smoothing_initialization(smoothing);
        model.set_smoothing_regression(smoothing);
        model.set_full_functional(full_functional);

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
        DMatrix<double> W{model.W()};
        DMatrix<double> V{model.V()};
        DMatrix<double> T{model.T()};
        DMatrix<double> C{model.C()};
        DMatrix<double> D{model.D()};
        DMatrix<double> Y_hat1{(T * D.transpose()).rowwise() + model.Y_mean().transpose()};
        DMatrix<double> Y_hat2{model.fitted()};
        DMatrix<double> Y_mean{model.Y_mean().transpose()};
        DMatrix<double> X_hat{model.reconstructed_field()};
        DMatrix<double> X_mean{model.X_mean().transpose()};
        DMatrix<double> B_hat{model.B()};

        Test_fPLSR::compare_NIPALS(test_directory, method_name, VERBOSE, false, W, V, T, C, D, Y_hat1, Y_hat2, Y_mean, X_hat, X_mean, B_hat);
    }
}

/* test 2:
   Checking that the functional method is consistent with the
   multivariate one for small values of lambda
   approach:     NIPALS
   domain:       unit square [0,1] x [0,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   BC:           no
   order FE:     1
 */
TEST(FPLSR, Test2_consistency_with_multivariate_NIPALS)
{
    // define domain and regularizing PDE
    MeshLoader<Mesh2D<>> domain("unit_square");
    auto L = Laplacian();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements() * 3, 1);
    PDE problem(domain.mesh, L, u);

    // define statistical model
    FPLSR<decltype(problem),
          SpaceOnly,
          fdaPDE::models::GeoStatMeshNodes,
          fdaPDE::models::fixed_lambda>
        model(problem);

    // tests
    std::string tests_directory = "data/models/FPLSR/2D_test/";
    std::string method_name = "nipals";
    bool VERBOSE = false;
    std::vector<unsigned int> tests{1, 2, 3, 4, 5};

    // multivariate model (NIPALS)
    std::vector<SVector<1>> lambdas{SVector<1>{1e-15}};
    bool full_functional = false;

    for (unsigned int i : tests)
    {

        if (VERBOSE)
        {
            std::cout << "##########" << std::endl;
            std::cout << "# Test " << i << " #" << std::endl;
            std::cout << "##########" << std::endl;
        }

        std::string test_directory = tests_directory + "test_" + std::to_string(i) + "/";

        // load data from .csv files
        CSVReader<double> reader{};
        CSVFile<double> yFile;
        yFile = reader.parseFile(test_directory + "Y.csv");
        DMatrix<double> Y = yFile.toEigen();
        CSVFile<double> xFile;
        xFile = reader.parseFile(test_directory + "X.csv");
        DMatrix<double> X = xFile.toEigen();

        // set model options
        model.set_verbose(VERBOSE);
        model.setLambda(lambdas);
        model.set_smoothing_initialization(false);
        model.set_smoothing_regression(true, lambdas);
        model.set_full_functional(full_functional);

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
        DMatrix<double> W{model.W()};
        DMatrix<double> V{model.V()};
        DMatrix<double> T{model.T()};
        DMatrix<double> C{model.C()};
        DMatrix<double> D{model.D()};
        DMatrix<double> Y_hat1{(T * D.transpose()).rowwise() + model.Y_mean().transpose()};
        DMatrix<double> Y_hat2{model.fitted()};
        DMatrix<double> Y_mean{model.Y_mean().transpose()};
        DMatrix<double> X_hat{model.reconstructed_field()};
        DMatrix<double> X_mean{model.X_mean().transpose()};
        DMatrix<double> B_hat{model.B()};

        Test_fPLSR::compare_NIPALS(test_directory, method_name, VERBOSE, true, W, V, T, C, D, Y_hat1, Y_hat2, Y_mean, X_hat, X_mean, B_hat);
    }
}

/* test 3:
   GCV selection approach for model hyperparameters
   approach:     NIPALS
   domain:       unit square [0,1] x [0,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   BC:           no
   order FE:     1
 */
TEST(FPLSR, Test3_GCV)
{
    // define domain and regularizing PDE
    MeshLoader<Mesh2D<>> domain("unit_square");
    auto L = Laplacian();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements() * 3, 1);
    PDE problem(domain.mesh, L, u);

    // define statistical model
    FPLSR<decltype(problem),
          SpaceOnly,
          fdaPDE::models::GeoStatMeshNodes,
          fdaPDE::models::fixed_lambda>
        model(problem);

    // tests
    std::string tests_directory = "data/models/FPLSR/2D_test/";
    std::string method_name = "nipals";
    bool VERBOSE = false;
    std::vector<unsigned int> tests{1};

    // multivariate model (NIPALS)
    std::vector<SVector<1>> lambdas;
    for (double x = -4; x <= 0; x += 1)
        lambdas.push_back(SVector<1>(std::pow(10, x)));
    bool full_functional = false;

    for (unsigned int i : tests)
    {

        if (VERBOSE)
        {
            std::cout << "##########" << std::endl;
            std::cout << "# Test " << i << " #" << std::endl;
            std::cout << "##########" << std::endl;
        }

        std::string test_directory = tests_directory + "test_" + std::to_string(i) + "/";

        // load data from .csv files
        CSVReader<double> reader{};
        CSVFile<double> yFile;
        yFile = reader.parseFile(test_directory + "Y.csv");
        DMatrix<double> Y = yFile.toEigen();
        CSVFile<double> xFile;
        xFile = reader.parseFile(test_directory + "X.csv");
        DMatrix<double> X = xFile.toEigen();

        // set model options
        model.set_verbose(VERBOSE);
        model.setLambda(lambdas);
        model.set_smoothing_initialization(true, lambdas);
        model.set_smoothing_regression(true, lambdas);
        model.set_full_functional(full_functional);

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
        DMatrix<double> W{model.W()};
        DMatrix<double> V{model.V()};
        DMatrix<double> T{model.T()};
        DMatrix<double> C{model.C()};
        DMatrix<double> D{model.D()};
        DMatrix<double> Y_hat1{(T * D.transpose()).rowwise() + model.Y_mean().transpose()};
        DMatrix<double> Y_hat2{model.fitted()};
        DMatrix<double> Y_mean{model.Y_mean().transpose()};
        DMatrix<double> X_hat{model.reconstructed_field()};
        DMatrix<double> X_mean{model.X_mean().transpose()};
        DMatrix<double> B_hat{model.B()};

        // Test_fPLSR::compare_NIPALS(test_directory, method_name, VERBOSE, true, W, V, T, C, D, Y_hat1, Y_hat2, Y_mean, X_hat, X_mean, B_hat);
    }
}

/* test 2:
   Checking that the functional method is consistent with the
   multivariate one for small values of lambda
   approach:     SIMPLS
   domain:       unit square [0,1] x [0,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   BC:           no
   order FE:     1
 */
/*
TEST(FPLSR, Test2_consistency_with_multivariate_SIMPLS)
{
    // define domain and regularizing PDE
    MeshLoader<Mesh2D<>> domain("unit_square");
    auto L = Laplacian();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements() * 3, 1);
    PDE problem(domain.mesh, L, u);

    // define statistical model
    FPLSR_SIMPLS<decltype(problem),
                 SpaceOnly,
                 fdaPDE::models::GeoStatMeshNodes,
                 fdaPDE::models::fixed_lambda>
        model(problem);

    // tests
    std::string tests_directory = "data/models/FPLSR/2D_test/";
    std::string method_name = "simpls";
    bool VERBOSE = false;
    std::vector<unsigned int> tests{1, 2, 3, 4, 5};

    // multivariate model (SIMPL)
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

        std::string test_directory = tests_directory + "test_" + std::to_string(i) + "/";

        // load data from .csv files
        CSVReader<double> reader{};
        CSVFile<double> yFile;
        yFile = reader.parseFile(test_directory + "Y.csv");
        DMatrix<double> Y = yFile.toEigen();
        CSVFile<double> xFile;
        xFile = reader.parseFile(test_directory + "X.csv");
        DMatrix<double> X = xFile.toEigen();

        // set model options
        model.setLambdaS(lambda);
        model.set_smoothing(smoothing);
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

        Test_fPLSR::compare_SIMPLS(test_directory, method_name, VERBOSE, false, R, Q, T, U, P, V, Y_hat, Y_mean, X_hat, X_mean, B_hat, "");
    }
}
*/