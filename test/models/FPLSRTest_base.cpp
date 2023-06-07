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

namespace Test_fPLSR_base
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

    file = reader.parseFile(test_directory + "Y_hat" + type + ".csv");
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

      file = reader.parseFile(test_directory + "W" + type + ".csv");
      DMatrix<double> expected_W = file.toEigen();
      if (VERBOSE)
      {
        std::cout << std::endl;
        std::cout << "||||||| W ||||||||:" << std::endl;
        std::cout << "Expected:" << std::endl;
        std::cout << expected_W.topRows(5) << std::endl;
        std::cout << "Obtained:" << std::endl;
        std::cout << W.topRows(5) << std::endl;
        std::cout << "Error norm: " << (Test_fPLSR_base::ns(expected_W) - Test_fPLSR_base::ns(W)).lpNorm<Eigen::Infinity>() << std::endl;
        std::cout << "----------------" << std::endl;
        std::cout << std::endl;
      }
      EXPECT_TRUE(almost_equal(ns(expected_W), ns(W)));

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
        std::cout << "Error norm: " << (Test_fPLSR_base::ns(expected_V) - Test_fPLSR_base::ns(V)).lpNorm<Eigen::Infinity>() << std::endl;
        std::cout << "----------------" << std::endl;
        std::cout << std::endl;
      }
      EXPECT_TRUE(almost_equal(ns(expected_V), ns(V)));

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
        std::cout << "Error norm: " << (Test_fPLSR_base::ns(expected_T) - Test_fPLSR_base::ns(T)).lpNorm<Eigen::Infinity>() << std::endl;
        std::cout << "----------------" << std::endl;
        std::cout << std::endl;
      }
      EXPECT_TRUE(almost_equal(ns(expected_T), ns(T)));

      file = reader.parseFile(test_directory + "C" + type + ".csv");
      DMatrix<double> expected_C = file.toEigen();
      if (VERBOSE)
      {
        std::cout << std::endl;
        std::cout << "||||||| C ||||||||:" << std::endl;
        std::cout << "Expected:" << std::endl;
        std::cout << expected_C.topRows(5) << std::endl;
        std::cout << "Obtained:" << std::endl;
        std::cout << C.topRows(5) << std::endl;
        std::cout << "Error norm: " << (Test_fPLSR_base::ns(expected_C) - Test_fPLSR_base::ns(C)).lpNorm<Eigen::Infinity>() << std::endl;
        std::cout << "----------------" << std::endl;
        std::cout << std::endl;
      }
      EXPECT_TRUE(almost_equal(ns(expected_C), ns(C)));

      file = reader.parseFile(test_directory + "D" + type + ".csv");
      DMatrix<double> expected_D = file.toEigen();
      if (VERBOSE)
      {
        std::cout << std::endl;
        std::cout << "||||||| D ||||||||:" << std::endl;
        std::cout << "Expected:" << std::endl;
        std::cout << expected_D.topRows(1) << std::endl;
        std::cout << "Obtained:" << std::endl;
        std::cout << D.topRows(1) << std::endl;
        std::cout << "Error norm: " << (Test_fPLSR_base::ns(expected_D) - Test_fPLSR_base::ns(D)).lpNorm<Eigen::Infinity>() << std::endl;
        std::cout << "----------------" << std::endl;
        std::cout << std::endl;
      }
      EXPECT_TRUE(almost_equal(ns(expected_D), ns(D)));
    }
  }

}

/* test 0:
   comparison with Harold's results
   domain:       unit square [0,1] x [0,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   BC:           no
   order FE:     1
 */
TEST(FPLSR, Test0_Laplacian_GeostatisticalAtNodes_comparison_with_harold)
{
  // define domain and regularizing PDE
  MeshLoader<Mesh2D<>> domain("unit_square");
  auto L = Laplacian();
  DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements() * 3, 1);
  PDE problem(domain.mesh, L, u); // definition of regularizing PDE

  // define statistical model
  FPLSR<decltype(problem),
        SpaceOnly,
        fdaPDE::models::GeoStatMeshNodes,
        fdaPDE::models::fixed_lambda>
      model(problem);

  // Tests
  std::string tests_directory = "data/models/FPLSR/2D_test0/";
  bool VERBOSE = false;
  std::vector<unsigned int> tests{1, 2, 3, 4, 5, 6};

  // Harold's model
  double lambda = 10;
  bool full_funtional = true;
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

    Test_fPLSR_base::compare(test_directory, VERBOSE, false, W, V, T, C, D, Y_hat1, Y_hat2, Y_mean, X_hat, X_mean, B_hat);
  }
}

TEST(FPLSR, Test0_Laplacian_GeostatisticalAtNodes_comparison_with_multivariate_NIPALS)
{
  // define domain and regularizing PDE
  MeshLoader<Mesh2D<>> domain("unit_square");
  auto L = Laplacian();
  DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements() * 3, 1);
  PDE problem(domain.mesh, L, u); // definition of regularizing PDE

  // define statistical model
  FPLSR<decltype(problem),
        SpaceOnly,
        fdaPDE::models::GeoStatMeshNodes,
        fdaPDE::models::fixed_lambda>
      model(problem);

  // Tests
  std::string tests_directory = "data/models/FPLSR/2D_test0/";
  bool VERBOSE = false;
  std::vector<unsigned int> tests{1, 2, 3, 4, 5, 6};

  // Multivariate model
  double lambda = 1e-14;
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

    Test_fPLSR_base::compare(test_directory, VERBOSE, true, W, V, T, C, D, Y_hat1, Y_hat2, Y_mean, X_hat, X_mean, B_hat, "_multivariate_NIPALS");
  }
}