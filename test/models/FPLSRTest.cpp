#include <cstddef>
#include <gtest/gtest.h> // testing framework
#include <unsupported/Eigen/SparseExtra>

#include "../fdaPDE/core/utils/Symbols.h"
#include "../fdaPDE/core/utils/IO/CSVReader.h"
#include "../fdaPDE/core/FEM/PDE.h"
using fdaPDE::core::FEM::PDE;
// #include "../fdaPDE/core/FEM/EigenValueProblem.h"
// using fdaPDE::core::FEM::EigenValueProblem;
#include "core/MESH/Mesh.h"
#include "../fdaPDE/models/functional/fPLSR.h"
using fdaPDE::models::FPLSR;
#include "../fdaPDE/models/SamplingDesign.h"
using fdaPDE::models::Sampling;
#include "../../fdaPDE/models/ModelTraits.h"
using fdaPDE::models::SolverType;
using fdaPDE::models::SpaceOnly;

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
const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "\n");

// #include "../fdaPDE/core/OPT/optimizers/Grid.h"

/* test 1
   domain:       unit square [1,1] x [1,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   BC:           no
   order FE:     1
 */

/*
BlockFrame<double, int> generate_2D_data(auto model, std::size_t n_samples, std::size_t l = 1, double Rsq = 0.95)
{

  // initilaization of the pde -> R0
  model.init_pde();

  // random engine
  std::default_random_engine engine(1);
  std::normal_distribution<double> distribution(0.0, 0.2);

  // get space for the data
  DMatrix<double> X;
  X.resize(n_samples, model.domain().nodes());
  DVector<double> beta;
  beta.resize(model.domain().nodes(), 1);
  DMatrix<double> nodes;
  nodes.resize(model.domain().nodes(), 2);

  // nodes
  for (std::size_t j = 0; j < model.domain().nodes(); ++j)
  {
    nodes.row(j) = model.domain().node(j);
  }

  // X
  for (std::size_t i = 0; i < n_samples; ++i)
  {

    const double a1 = 1 + distribution(engine);
    const double a2 = 1 + distribution(engine);

    for (std::size_t j = 0; j < model.domain().nodes(); ++j)
    {

      X(i, j) = a1 * std::cos(2 * M_PI * nodes(j, 0)) +
                a2 * std::cos(2 * M_PI * nodes(j, 1)) +
                1 + distribution(engine);
    }
  }
  DVector<double> X_mean{X.colwise().mean()};
  DMatrix<double> Xc{X.rowwise() - X_mean.transpose()};

  // Beta 1
  const double r = 0.4; // set the r parameter
  beta = 5 * (nodes.col(0).unaryExpr([](const double i)
                                     { return (i - 0.5) * (i - 0.5); }) +
              nodes.col(1).unaryExpr([](const double i)
                                     { return (i - 0.5) * (i - 0.5); }))
                 .unaryExpr([r](const double i)
                            { return std::exp(-i / (2 * r * r)); });

  // Y
  DVector<double> Y_clean{Xc * model.R0() * beta};
  const double Y_clean_var = (Y_clean.array() - Y_clean.mean()).matrix().squaredNorm() / static_cast<double>(Y_clean.rows() - 1);
  const double var_e = (1 / Rsq - 1) * Y_clean_var;

  std::normal_distribution<double> distribution_error(0.0, var_e);

  DVector<double> Y = Y_clean.unaryExpr([&engine, &distribution_error](const double i)
                                        { return i + distribution_error(engine); });

  std::ofstream Xfile;
  Xfile.open("data/models/FPLSR/2D_test1/myX.csv");
  Xfile << X.format(CSVFormat);
  Xfile.close();

  std::ofstream nodesfile;
  nodesfile.open("data/models/FPLSR/2D_test1/nodes.csv");
  nodesfile << nodes.format(CSVFormat);
  nodesfile.close();

  std::ofstream betafile;
  betafile.open("data/models/FPLSR/2D_test1/beta.csv");
  betafile << beta.format(CSVFormat);
  betafile.close();

  std::ofstream Yfile;
  Yfile.open("data/models/FPLSR/2D_test1/myY_clean.csv");
  Yfile << Y.format(CSVFormat);
  Yfile.close();

  std::ofstream Ycleanfile;
  Ycleanfile.open("data/models/FPLSR/2D_test1/myY.csv");
  Ycleanfile << Y_clean.format(CSVFormat);
  Ycleanfile.close();

  BlockFrame<double, int> df_data;

  df_data.insert<double>(OBSERVATIONS_BLK, Y);
  df_data.insert<double>(DESIGN_MATRIX_BLK, X);

  return df_data;

  // std::cout << X.rows() << " " << X.cols() << std::endl;
  // std::cout << Y.rows() << " " << Y.cols() << std::endl;
}
*/

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

TEST(FPLSR, Test1_Laplacian_GeostatisticalAtNodes)
{
  // define domain and regularizing PDE
  MeshLoader<Mesh2D<>> domain("unit_square");
  auto L = Laplacian();
  DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements() * 3, 1);
  PDE problem(domain.mesh, L, u); // definition of regularizing PDE

  // define statistical model
  FPLSR<decltype(problem), SpaceOnly, fdaPDE::models::Sampling::GeoStatMeshNodes,
        fdaPDE::models::fixed_lambda>
      model(problem);

  // Tests
  std::string tests_directory = "data/models/FPLSR/2D_test1/";
  bool VERBOSE = false;
  std::vector<unsigned int> tests{1, 2, 3, 4, 5, 6};

  for (unsigned int i : tests)
  {

    if (VERBOSE)
    {
      std::cout << "##########" << std::endl;
      std::cout << "# Test " << i << "#" << std::endl;
      std::cout << "##########" << std::endl;
    }

    std::string test_directory = tests_directory + "test" + std::to_string(i) + "/";

    // BlockFrame<double, int> df_data{generate_2D_data(model, 50, 2)};

    // load data from .csv files
    CSVReader<double> reader{};
    CSVFile<double> yFile; // observation file
    CSVFile<double> xFile; // covariates file

    yFile = reader.parseFile(test_directory + "Y.csv");
    xFile = reader.parseFile(test_directory + "X.csv");
    DMatrix<double> Y = yFile.toEigen();
    DMatrix<double> X = xFile.toEigen();

    // set model data
    BlockFrame<double, int> df_data;
    df_data.insert(OBSERVATIONS_BLK, DMatrix<double>(Y));
    df_data.insert(DESIGN_MATRIX_BLK, DMatrix<double>(X));

    model.setData(df_data);

    // smoothing parameter
    double lambda = 10;
    model.setLambdaS(lambda);
    // std::vector<SVector<1>> lambdas;
    // for (double x = -6.0; x <= -2.0; x++)
    //   lambdas.push_back(SVector<1>(std::pow(10, x)));

    // solve smoothing problem
    model.init();
    model.solve();

    // Results
    DMatrix<double> W{model.W()};
    DMatrix<double> T{model.T()};
    DMatrix<double> C{model.C()};
    DMatrix<double> D{model.D()};
    DMatrix<double> Y_hat1{(T * D.transpose()).rowwise() + model.Y_mean().transpose()};
    DMatrix<double> Y_hat2{model.fitted()};
    DMatrix<double> B_hat{model.B()};

    //   **  test correctness of computed results  **

    CSVFile<double> file; // covariates file

    file = reader.parseFile(test_directory + "Y_hat.csv");
    DMatrix<double> expected_Y_hat = file.toEigen();
    if (VERBOSE)
    {
      std::cout << "Expected:" << std::endl;
      std::cout << expected_Y_hat.topRows(5) << std::endl;
      std::cout << "Obtained version 1:" << std::endl;
      std::cout << Y_hat1.topRows(5) << std::endl;
      std::cout << "Obtained version 2:" << std::endl;
      std::cout << Y_hat2.topRows(5) << std::endl;
      std::cout << "Error norm 1: " << (expected_Y_hat - Y_hat1).lpNorm<Eigen::Infinity>() << std::endl;
      std::cout << "Error norm 2: " << (expected_Y_hat - Y_hat2).lpNorm<Eigen::Infinity>() << std::endl;
      std::cout << "----------------" << std::endl;
    }
    EXPECT_TRUE(almost_equal(expected_Y_hat, Y_hat1));
    EXPECT_TRUE(almost_equal(expected_Y_hat, Y_hat2));

    file = reader.parseFile(test_directory + "B_hat.csv");
    DMatrix<double> expected_B_hat = file.toEigen();
    if (VERBOSE)
    {
      std::cout << "Expected:" << std::endl;
      std::cout << expected_B_hat.topRows(5) << std::endl;
      std::cout << "Obtained:" << std::endl;
      std::cout << B_hat.topRows(5) << std::endl;
      std::cout << "Error norm: " << (expected_B_hat - B_hat).lpNorm<Eigen::Infinity>() << std::endl;
      std::cout << "----------------" << std::endl;
    }
    EXPECT_TRUE(almost_equal(expected_B_hat, B_hat));

    file = reader.parseFile(test_directory + "W.csv");
    DMatrix<double> expected_W = file.toEigen();
    if (VERBOSE)
    {
      std::cout << "Expected:" << std::endl;
      std::cout << expected_W.topRows(5) << std::endl;
      std::cout << "Obtained:" << std::endl;
      std::cout << W.topRows(5) << std::endl;
      std::cout << "Error norm: " << (ns(expected_W) - ns(W)).lpNorm<Eigen::Infinity>() << std::endl;
      std::cout << "----------------" << std::endl;
    }
    EXPECT_TRUE(almost_equal(ns(expected_W), ns(W)));

    file = reader.parseFile(test_directory + "T.csv");
    DMatrix<double> expected_T = file.toEigen();
    if (VERBOSE)
    {
      std::cout << "Expected:" << std::endl;
      std::cout << expected_T.topRows(5) << std::endl;
      std::cout << "Obtained:" << std::endl;
      std::cout << T.topRows(5) << std::endl;
      std::cout << "Error norm: " << (ns(expected_T) - ns(T)).lpNorm<Eigen::Infinity>() << std::endl;
      std::cout << "----------------" << std::endl;
    }
    EXPECT_TRUE(almost_equal(ns(expected_T), ns(T)));

    file = reader.parseFile(test_directory + "C.csv");
    DMatrix<double> expected_C = file.toEigen();
    if (VERBOSE)
    {
      std::cout << "Expected:" << std::endl;
      std::cout << expected_C.topRows(5) << std::endl;
      std::cout << "Obtained:" << std::endl;
      std::cout << C.topRows(5) << std::endl;
      std::cout << "Error norm: " << (ns(expected_C) - ns(C)).lpNorm<Eigen::Infinity>() << std::endl;
      std::cout << "----------------" << std::endl;
    }
    EXPECT_TRUE(almost_equal(ns(expected_C), ns(C)));

    file = reader.parseFile(test_directory + "D.csv");
    DMatrix<double> expected_D = file.toEigen();
    if (VERBOSE)
    {
      std::cout << "Expected:" << std::endl;
      std::cout << expected_D.topRows(1) << std::endl;
      std::cout << "Obtained:" << std::endl;
      std::cout << D.topRows(1) << std::endl;
      std::cout << "Error norm: " << (ns(expected_D) - ns(D)).lpNorm<Eigen::Infinity>() << std::endl;
      std::cout << "----------------" << std::endl;
    }
    EXPECT_TRUE(almost_equal(ns(expected_D), ns(D)));
  }
}

TEST(FPLSR, Test1_Laplacian_GeostatisticalAtNodes_Comparison)
{
  // define domain and regularizing PDE
  MeshLoader<Mesh2D<>> domain("unit_square");
  auto L = Laplacian();
  DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements() * 3, 1);
  PDE problem(domain.mesh, L, u); // definition of regularizing PDE

  // define statistical model
  FPLSR<decltype(problem), SpaceOnly, fdaPDE::models::Sampling::GeoStatMeshNodes,
        fdaPDE::models::fixed_lambda>
      model(problem);

  // tests
  std::string tests_directory = "data/models/FPLSR/2D_test1/";
  std::string method_name = "";
  std::string comparison_directory = tests_directory + "comparison/";
  if (!std::filesystem::exists(comparison_directory))
    std::filesystem::create_directory(comparison_directory);

  bool VERBOSE = false;
  std::vector<unsigned int> tests{1, 2, 3, 4, 5, 6};

  // room to store the errors
  std::vector<double> errors_Y;
  std::vector<double> errors_X;
  std::vector<double> errors_B;
  errors_Y.reserve(tests.size());
  errors_X.reserve(tests.size());
  errors_B.reserve(tests.size());

  // output file
  std::ofstream outfile;

  for (unsigned int i : tests)
  {

    if (VERBOSE)
    {
      std::cout << "##########" << std::endl;
      std::cout << "# Test " << i << "#" << std::endl;
      std::cout << "##########" << std::endl;
    }

    // directories
    std::string test_directory = tests_directory + "test" + std::to_string(i) + "/";
    std::string comparison_test_directory = comparison_directory + "test" + std::to_string(i) + "/";
    if (!std::filesystem::exists(comparison_test_directory))
      std::filesystem::create_directory(comparison_test_directory);

    // load data from .csv files
    CSVReader<double> reader{};
    CSVFile<double> yFile; // observation file
    CSVFile<double> xFile; // covariates file
    yFile = reader.parseFile(test_directory + "Y.csv");
    xFile = reader.parseFile(test_directory + "X.csv");
    DMatrix<double> Y = yFile.toEigen();
    DMatrix<double> X = xFile.toEigen();

    // set model data
    BlockFrame<double, int> df_data;
    df_data.insert(OBSERVATIONS_BLK, DMatrix<double>(Y));
    df_data.insert(DESIGN_MATRIX_BLK, DMatrix<double>(X));

    model.setData(df_data);

    // smoothing parameter
    double lambda = 10;
    model.setLambdaS(lambda);
    // std::vector<SVector<1>> lambdas;
    // for (double x = -6.0; x <= -2.0; x++)
    //   lambdas.push_back(SVector<1>(std::pow(10, x)));

    // solve smoothing problem
    model.init();
    model.solve();

    // Results
    DMatrix<double> Y_hat{model.fitted()};
    DMatrix<double> X_hat{model.reconstructedField()};
    DMatrix<double> B_hat{model.B()};

    //   **  comparison with original data  **

    CSVFile<double> file; // covariates file

    file = reader.parseFile(test_directory + "Y_clean.csv");
    DMatrix<double> Y_clean = file.toEigen();
    errors_Y.push_back((Y_clean - Y_hat).lpNorm<Eigen::Infinity>());
    if (VERBOSE)
    {
      std::cout << "Expected:" << std::endl;
      std::cout << Y_clean.topRows(5) << std::endl;
      std::cout << "Obtained version:" << std::endl;
      std::cout << Y_hat.topRows(5) << std::endl;
      std::cout << "Error norm 1 " << errors_Y.back() << std::endl;
      std::cout << "----------------" << std::endl;
    }
    outfile.open(comparison_test_directory + "Y_hat_" + method_name + ".csv");
    outfile << Y_hat.format(CSVFormat);
    outfile.close();

    file = reader.parseFile(test_directory + "X_clean.csv");
    DMatrix<double> X_clean = file.toEigen();
    errors_X.push_back((X_clean - X_hat).lpNorm<Eigen::Infinity>());
    if (VERBOSE)
    {
      std::cout << "Expected:" << std::endl;
      std::cout << X_clean.topRows(5) << std::endl;
      std::cout << "Obtained version:" << std::endl;
      std::cout << X_hat.topRows(5) << std::endl;
      std::cout << "Error norm 1 " << errors_X.back() << std::endl;
      std::cout << "----------------" << std::endl;
    }
    outfile.open(comparison_test_directory + "X_hat_" + method_name + ".csv");
    outfile << X_hat.format(CSVFormat);
    outfile.close();

    file = reader.parseFile(test_directory + "B.csv");
    DMatrix<double> B = file.toEigen();
    errors_B.push_back((B - B_hat).lpNorm<Eigen::Infinity>());
    if (VERBOSE)
    {
      std::cout << "Expected:" << std::endl;
      std::cout << B.topRows(5) << std::endl;
      std::cout << "Obtained version:" << std::endl;
      std::cout << B_hat.topRows(5) << std::endl;
      std::cout << "Error norm 1 " << errors_B.back() << std::endl;
      std::cout << "----------------" << std::endl;
    }
    outfile.open(comparison_test_directory + "B_hat_" + method_name + ".csv");
    outfile << B_hat.format(CSVFormat);
    outfile.close();
  }

  std::ofstream results(comparison_directory + "results_" + method_name + ".csv");

  if (VERBOSE)
  {
    std::cout << "Results: " << std::endl;
    std::cout << std::setw(10) << std::left << "Tests"
              << std::setw(12) << std::right << "Y_error"
              << std::setw(12) << std::right << "X_error"
              << std::setw(12) << std::right << "B_error" << std::endl;
  }
  results << "\"Test\",\"Y_error\",\"X_error\",\"B_error\"" << std::endl;
  for (unsigned int i : tests)
  {
    if (VERBOSE)
    {
      std::string test_name = "Test " + std::to_string(i) + ":";
      std::cout << std::setw(10) << std::left << test_name << std::right
                << std::setw(12) << errors_Y[i - 1]
                << std::setw(12) << errors_X[i - 1]
                << std::setw(12) << errors_B[i - 1] << std::endl;
    }
    results << "\"Test" << i << "\","
            << errors_Y[i - 1] << ","
            << errors_X[i - 1] << ","
            << errors_B[i - 1] << std::endl;
  }
  results.close();
}