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

const static Eigen::IOFormat CSVFormat1(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "\n");

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

/* test 0:
   comparison with Harold's results
   domain:       unit square [0,1] x [0,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   BC:           no
   order FE:     1
 */
/*
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
        fdaPDE::models::Sampling::GeoStatMeshNodes,
        fdaPDE::models::fixed_lambda>
      model(problem);

  // Tests
  std::string tests_directory = "data/models/FPLSR/2D_test0/";
  bool VERBOSE = false;
  std::vector<unsigned int> tests{1, 2, 3, 4, 5, 6};

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

    // set model data
    BlockFrame<double, int> df_data;
    df_data.insert(OBSERVATIONS_BLK, Y);
    df_data.insert(DESIGN_MATRIX_BLK, X);
    model.setDataExtra(df_data);
    model.setData(df_data);

    // set smoothing parameter
    double lambda = 10;
    model.setLambdaS(lambda);

    // solve smoothing problem
    model.init();
    model.solve();

    //   **  test correctness of computed results  **   //

    // Results
    DMatrix<double> W{model.W()};
    DMatrix<double> T{model.T()};
    DMatrix<double> C{model.C()};
    DMatrix<double> D{model.D()};
    DMatrix<double> Y_hat1{(T * D.transpose()).rowwise() + model.Y_mean().transpose()};
    DMatrix<double> Y_hat2{model.fitted()};
    DMatrix<double> B_hat{model.B()};
    CSVFile<double> file;
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
*/

/* test 1:
   domain:       unit square [0,1] x [0,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   BC:           no
   order FE:     1
 */
TEST(FPLSR, Test1_Laplacian_GeostatisticalAtNodes)
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

  // tests
  std::string tests_directory = "data/models/FPLSR/2D_test1/";

  bool VERBOSE = false;
  std::vector<unsigned int> tests{1, 2, 3, 4, 5, 6};

  // room to store the errors
  std::vector<double> errors_Y;
  std::vector<double> errors_X;
  std::vector<double> errors_B;
  errors_Y.reserve(tests.size());
  errors_X.reserve(tests.size());
  errors_B.reserve(tests.size());

  // reader
  CSVReader<double> reader{};

  // output file
  std::ofstream outfile;

  for (unsigned int i : tests)
  {

    if (VERBOSE)
    {
      std::cout << "##########" << std::endl;
      std::cout << "# Test " << i << " #" << std::endl;
      std::cout << "##########" << std::endl;
    }

    // directories
    std::string test_directory = tests_directory + "test" + std::to_string(i) + "/";
    std::string results_directory = test_directory + "results/";
    if (!std::filesystem::exists(results_directory))
      std::filesystem::create_directory(results_directory);

    // smoothing parameter
    double lambda = 10;
    model.setLambdaS(lambda);

    // set number of latent components
    // model.set_H(3);

    // load data from .csv files
    CSVFile<double> yFile; // observation file
    yFile = reader.parseFile(test_directory + "Y.csv");
    DMatrix<double> Y = yFile.toEigen();
    CSVFile<double> xFile; // covariates file
    xFile = reader.parseFile(test_directory + "X.csv");
    DMatrix<double> X = xFile.toEigen();

    // set model data
    BlockFrame<double, int> df_data;
    df_data.insert(OBSERVATIONS_BLK, DMatrix<double>(Y));
    df_data.insert(DESIGN_MATRIX_BLK, DMatrix<double>(X));
    model.setData(df_data);

    // solve smoothing problem
    model.init();
    model.solve();

    // Results
    DMatrix<double> Y_hat{model.fitted()};
    DMatrix<double> X_hat{model.reconstructed_field()};
    DMatrix<double> B_hat{model.B()};

    //   **  compare and export results  **   //

    CSVFile<double> file; // covariates file

    // Y
    file = reader.parseFile(test_directory + "Y_clean.csv");
    DMatrix<double> Y_clean = file.toEigen();
    errors_Y.push_back((Y_clean - Y_hat).lpNorm<Eigen::Infinity>());
    if (VERBOSE)
    {
      std::cout << "Expected:" << std::endl;
      std::cout << Y_clean.topRows(5) << std::endl;
      std::cout << "Obtained version:" << std::endl;
      std::cout << Y_hat.topRows(5) << std::endl;
      std::cout << "Error norm " << errors_Y.back() << std::endl;
      std::cout << "----------------" << std::endl;
    }
    outfile.open(results_directory + "Y_hat.csv");
    outfile << Y_hat.format(CSVFormat1);
    outfile.close();

    // X
    file = reader.parseFile(test_directory + "X_clean.csv");
    DMatrix<double> X_clean = file.toEigen();
    errors_X.push_back((X_clean - X_hat).lpNorm<Eigen::Infinity>());
    if (VERBOSE)
    {
      // std::cout << "Expected:" << std::endl;
      // std::cout << X_clean.topRows(5) << std::endl;
      // std::cout << "Obtained version:" << std::endl;
      // std::cout << X_hat.topRows(5) << std::endl;
      std::cout << "Error norm " << errors_X.back() << std::endl;
      std::cout << "----------------" << std::endl;
    }
    outfile.open(results_directory + "X_hat.csv");
    outfile << X_hat.format(CSVFormat1);
    outfile.close();

    // B
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
    outfile.open(results_directory + "B_hat.csv");
    outfile << B_hat.format(CSVFormat1);
    outfile.close();
  }

  std::ofstream results(tests_directory + "errors.csv");

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

/* test 2:
   domain:       unit square [0,1] x [0,1]
   sampling:     locations != nodes, #locations == #nodes
   penalization: simple laplacian
   BC:           no
   order FE:     1
 */
TEST(FPLSR, Test2_Laplacian_AtLocations)
{

  // define domain and regularizing PDE
  MeshLoader<Mesh2D<>> domain("unit_square");
  auto L = Laplacian();
  DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements() * 3, 1);
  PDE problem(domain.mesh, L, u); // definition of regularizing PDE

  // define statistical model
  FPLSR<decltype(problem), SpaceOnly, fdaPDE::models::GeoStatLocations,
        fdaPDE::models::fixed_lambda>
      model(problem);

  // tests
  std::string tests_directory = "data/models/FPLSR/2D_test2/";

  bool VERBOSE = false;
  std::vector<unsigned int> tests{1, 2, 3, 4, 5, 6};

  // room to store the errors
  std::vector<double> errors_Y;
  std::vector<double> errors_X;
  std::vector<double> errors_B;
  errors_Y.reserve(tests.size());
  errors_X.reserve(tests.size());
  errors_B.reserve(tests.size());

  // reader
  CSVReader<double> reader{};

  // output file
  std::ofstream outfile;

  for (unsigned int i : tests)
  {

    if (VERBOSE)
    {
      std::cout << "##########" << std::endl;
      std::cout << "# Test " << i << " #" << std::endl;
      std::cout << "##########" << std::endl;
    }

    // directories
    std::string test_directory = tests_directory + "test" + std::to_string(i) + "/";
    std::string results_directory = test_directory + "results/";
    if (!std::filesystem::exists(results_directory))
      std::filesystem::create_directory(results_directory);

    // load locations from -csv files
    CSVFile<double> locFile;
    locFile = reader.parseFile(test_directory + "locations.csv");
    DMatrix<double> locs = locFile.toEigen();

    // set locations
    model.set_spatial_locations(locs);

    // set smoothing parameter
    double lambda = 10;
    model.setLambdaS(lambda);

    // set number of latent components
    // model.set_H(3);

    // load data from .csv files
    CSVFile<double> yFile; // observation file
    yFile = reader.parseFile(test_directory + "Y.csv");
    DMatrix<double> Y = yFile.toEigen();
    CSVFile<double> xFile; // covariates file
    xFile = reader.parseFile(test_directory + "X_locations.csv");
    DMatrix<double> X = xFile.toEigen();

    // set data
    BlockFrame<double, int> df_data;
    df_data.insert(OBSERVATIONS_BLK, DMatrix<double>(Y));
    df_data.insert(DESIGN_MATRIX_BLK, DMatrix<double>(X));
    model.setData(df_data);

    // solve the problem
    model.init();
    model.solve();

    // Results
    DMatrix<double> Y_hat{model.fitted()};
    DMatrix<double> X_hat{model.reconstructed_field()};
    DMatrix<double> B_hat{model.B()};

    //   **  compare and export results  **   //

    CSVFile<double> file; // covariates file

    // Y
    file = reader.parseFile(test_directory + "Y_clean.csv");
    DMatrix<double> Y_clean = file.toEigen();
    errors_Y.push_back((Y_clean - Y_hat).lpNorm<Eigen::Infinity>());
    if (VERBOSE)
    {
      std::cout << "Expected:" << std::endl;
      std::cout << Y_clean.topRows(5) << std::endl;
      std::cout << "Obtained version:" << std::endl;
      std::cout << Y_hat.topRows(5) << std::endl;
      std::cout << "Error norm " << errors_Y.back() << std::endl;
      std::cout << "----------------" << std::endl;
    }
    outfile.open(results_directory + "Y_hat.csv");
    outfile << Y_hat.format(CSVFormat1);
    outfile.close();

    // X
    file = reader.parseFile(test_directory + "X_clean.csv");
    DMatrix<double> X_clean = file.toEigen();
    errors_X.push_back((X_clean - X_hat).lpNorm<Eigen::Infinity>());
    if (VERBOSE)
    {
      // std::cout << "Expected:" << std::endl;
      // std::cout << X_clean.topRows(5) << std::endl;
      // std::cout << "Obtained version:" << std::endl;
      // std::cout << X_hat.topRows(5) << std::endl;
      std::cout << "Error norm " << errors_X.back() << std::endl;
      std::cout << "----------------" << std::endl;
    }
    outfile.open(results_directory + "X_hat.csv");
    outfile << X_hat.format(CSVFormat1);
    outfile.close();

    // B
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
    outfile.open(results_directory + "B_hat.csv");
    outfile << B_hat.format(CSVFormat1);
    outfile.close();
  }

  std::ofstream results(tests_directory + "errors.csv");

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

/* test 3:
   domain:       unit square [0,1] x [0,1]
   sampling:     locations != nodes, #locations < #nodes
   penalization: simple laplacian
   BC:           no
   order FE:     1
 */
TEST(FPLSR, Test3_Laplacian_AtLocations)
{

  // define domain and regularizing PDE
  MeshLoader<Mesh2D<>> domain("unit_square");
  auto L = Laplacian();
  DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements() * 3, 1);
  PDE problem(domain.mesh, L, u); // definition of regularizing PDE

  // define statistical model
  FPLSR<decltype(problem), SpaceOnly, fdaPDE::models::GeoStatLocations,
        fdaPDE::models::fixed_lambda>
      model(problem);

  // tests
  std::string tests_directory = "data/models/FPLSR/2D_test3/";

  bool VERBOSE = false;
  std::vector<unsigned int> tests{1, 2, 3, 4, 5, 6};

  // room to store the errors
  std::vector<double> errors_Y;
  std::vector<double> errors_X;
  std::vector<double> errors_B;
  errors_Y.reserve(tests.size());
  errors_X.reserve(tests.size());
  errors_B.reserve(tests.size());

  // reader
  CSVReader<double> reader{};

  // output file
  std::ofstream outfile;

  for (unsigned int i : tests)
  {

    if (VERBOSE)
    {
      std::cout << "##########" << std::endl;
      std::cout << "# Test " << i << " #" << std::endl;
      std::cout << "##########" << std::endl;
    }

    // directories
    std::string test_directory = tests_directory + "test" + std::to_string(i) + "/";
    std::string results_directory = test_directory + "results/";
    if (!std::filesystem::exists(results_directory))
      std::filesystem::create_directory(results_directory);

    // load locations from -csv files
    CSVFile<double> locFile;
    locFile = reader.parseFile(test_directory + "locations.csv");
    DMatrix<double> locs = locFile.toEigen();

    // set locations
    model.set_spatial_locations(locs);

    // set smoothing parameter
    double lambda = 10;
    model.setLambdaS(lambda);

    // set number of latent components
    // model.set_H(3);

    // load data from .csv files
    CSVFile<double> yFile; // observation file
    yFile = reader.parseFile(test_directory + "Y.csv");
    DMatrix<double> Y = yFile.toEigen();
    CSVFile<double> xFile; // covariates file
    xFile = reader.parseFile(test_directory + "X_locations.csv");
    DMatrix<double> X = xFile.toEigen();

    // set data
    BlockFrame<double, int> df_data;
    df_data.insert(OBSERVATIONS_BLK, DMatrix<double>(Y));
    df_data.insert(DESIGN_MATRIX_BLK, DMatrix<double>(X));
    model.setData(df_data);

    // solve the problem
    model.init();
    model.solve();

    // Results
    DMatrix<double> Y_hat{model.fitted()};
    DMatrix<double> X_hat{model.reconstructed_field()};
    DMatrix<double> B_hat{model.B()};

    //   **  compare and export results  **   //

    CSVFile<double> file; // covariates file

    // Y
    file = reader.parseFile(test_directory + "Y_clean.csv");
    DMatrix<double> Y_clean = file.toEigen();
    errors_Y.push_back((Y_clean - Y_hat).lpNorm<Eigen::Infinity>());
    if (VERBOSE)
    {
      std::cout << "Expected:" << std::endl;
      std::cout << Y_clean.topRows(5) << std::endl;
      std::cout << "Obtained version:" << std::endl;
      std::cout << Y_hat.topRows(5) << std::endl;
      std::cout << "Error norm " << errors_Y.back() << std::endl;
      std::cout << "----------------" << std::endl;
    }
    outfile.open(results_directory + "Y_hat.csv");
    outfile << Y_hat.format(CSVFormat1);
    outfile.close();

    // X
    file = reader.parseFile(test_directory + "X_clean.csv");
    DMatrix<double> X_clean = file.toEigen();
    errors_X.push_back((X_clean - X_hat).lpNorm<Eigen::Infinity>());
    if (VERBOSE)
    {
      // std::cout << "Expected:" << std::endl;
      // std::cout << X_clean.topRows(5) << std::endl;
      // std::cout << "Obtained version:" << std::endl;
      // std::cout << X_hat.topRows(5) << std::endl;
      std::cout << "Error norm " << errors_X.back() << std::endl;
      std::cout << "----------------" << std::endl;
    }
    outfile.open(results_directory + "X_hat.csv");
    outfile << X_hat.format(CSVFormat1);
    outfile.close();

    // B
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
    outfile.open(results_directory + "B_hat.csv");
    outfile << B_hat.format(CSVFormat1);
    outfile.close();
  }

  std::ofstream results(tests_directory + "errors.csv");

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

/* test 4:
   domain:       unit square [0,1] x [0,1]
   sampling:     locations != nodes, #locations << #nodes in a equispaced subgrid
   penalization: simple laplacian
   BC:           no
   order FE:     1
 */
TEST(FPLSR, Test4_Laplacian_AtLocations)
{

  // define domain and regularizing PDE
  MeshLoader<Mesh2D<>> domain("unit_square");
  auto L = Laplacian();
  DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements() * 3, 1);
  PDE problem(domain.mesh, L, u); // definition of regularizing PDE

  // define statistical model
  FPLSR<decltype(problem), SpaceOnly, fdaPDE::models::GeoStatLocations,
        fdaPDE::models::fixed_lambda>
      model(problem);

  // tests
  std::string tests_directory = "data/models/FPLSR/2D_test4/";

  bool VERBOSE = false;
  std::vector<unsigned int> tests{1, 2, 3, 4, 5, 6};

  // room to store the errors
  std::vector<double> errors_Y;
  std::vector<double> errors_X;
  std::vector<double> errors_B;
  errors_Y.reserve(tests.size());
  errors_X.reserve(tests.size());
  errors_B.reserve(tests.size());

  // reader
  CSVReader<double> reader{};

  // output file
  std::ofstream outfile;

  for (unsigned int i : tests)
  {

    if (VERBOSE)
    {
      std::cout << "##########" << std::endl;
      std::cout << "# Test " << i << " #" << std::endl;
      std::cout << "##########" << std::endl;
    }

    // directories
    std::string test_directory = tests_directory + "test" + std::to_string(i) + "/";
    std::string results_directory = test_directory + "results/";
    if (!std::filesystem::exists(results_directory))
      std::filesystem::create_directory(results_directory);

    // load locations from -csv files
    CSVFile<double> locFile;
    locFile = reader.parseFile(test_directory + "locations.csv");
    DMatrix<double> locs = locFile.toEigen();

    // set locations
    model.set_spatial_locations(locs);

    // set smoothing parameter
    double lambda = 10;
    model.setLambdaS(lambda);

    // set number of latent components
    // model.set_H(3);

    // load data from .csv files
    CSVFile<double> yFile; // observation file
    yFile = reader.parseFile(test_directory + "Y.csv");
    DMatrix<double> Y = yFile.toEigen();
    CSVFile<double> xFile; // covariates file
    xFile = reader.parseFile(test_directory + "X_locations.csv");
    DMatrix<double> X = xFile.toEigen();

    // set data
    BlockFrame<double, int> df_data;
    df_data.insert(OBSERVATIONS_BLK, DMatrix<double>(Y));
    df_data.insert(DESIGN_MATRIX_BLK, DMatrix<double>(X));
    model.setData(df_data);

    // solve the problem
    model.init();
    model.solve();

    // Results
    DMatrix<double> Y_hat{model.fitted()};
    DMatrix<double> X_hat{model.reconstructed_field()};
    DMatrix<double> B_hat{model.B()};

    //   **  compare and export results  **   //

    CSVFile<double> file; // covariates file

    // Y
    file = reader.parseFile(test_directory + "Y_clean.csv");
    DMatrix<double> Y_clean = file.toEigen();
    errors_Y.push_back((Y_clean - Y_hat).lpNorm<Eigen::Infinity>());
    if (VERBOSE)
    {
      std::cout << "Expected:" << std::endl;
      std::cout << Y_clean.topRows(5) << std::endl;
      std::cout << "Obtained version:" << std::endl;
      std::cout << Y_hat.topRows(5) << std::endl;
      std::cout << "Error norm " << errors_Y.back() << std::endl;
      std::cout << "----------------" << std::endl;
    }
    outfile.open(results_directory + "Y_hat.csv");
    outfile << Y_hat.format(CSVFormat1);
    outfile.close();

    // X
    file = reader.parseFile(test_directory + "X_clean.csv");
    DMatrix<double> X_clean = file.toEigen();
    errors_X.push_back((X_clean - X_hat).lpNorm<Eigen::Infinity>());
    if (VERBOSE)
    {
      // std::cout << "Expected:" << std::endl;
      // std::cout << X_clean.topRows(5) << std::endl;
      // std::cout << "Obtained version:" << std::endl;
      // std::cout << X_hat.topRows(5) << std::endl;
      std::cout << "Error norm " << errors_X.back() << std::endl;
      std::cout << "----------------" << std::endl;
    }
    outfile.open(results_directory + "X_hat.csv");
    outfile << X_hat.format(CSVFormat1);
    outfile.close();

    // B
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
    outfile.open(results_directory + "B_hat.csv");
    outfile << B_hat.format(CSVFormat1);
    outfile.close();
  }

  std::ofstream results(tests_directory + "errors.csv");

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