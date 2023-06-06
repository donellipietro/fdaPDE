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

namespace Test_fPLSR_comparison
{

  const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "\n");

}

/* test comparison:
   quantitative comparison with Harold's results
   domain:       unit square [0,1] x [0,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   BC:           no
   order FE:     1
 */
TEST(FPLSR, Test_comparison_Laplacian_GeostatisticalAtNodes)
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
  std::string tests_directory = "data/models/FPLSR/2D_test_comparison/";
  bool VERBOSE = true;
  std::vector<unsigned int> tests{1, 2, 3, 4, 5, 6};
  unsigned int n_batches = 10;
  unsigned int n_tests = tests.size();

  // reader
  CSVReader<double> reader{};

  // tests options
  bool TEST = true;
  std::vector<std::string> test_name_vect = {"hcpp_l0", "ns_l0", "hcpp", "ns", "sr", "sri"};
  unsigned int n_test_options = test_name_vect.size();
  std::vector<bool> smoothing_initialization_vect = {false, false, false, false, false, true};
  std::vector<bool> smoothing_regression_vect = {false, false, false, false, true, true};
  std::vector<bool> full_functional_vect = {true, false, true, false, false, false};
  std::vector<double> lambda_vect = {1e-12, 1e-12, 10, 10, 10, 10};

  // error matrices
  DMatrix<double> errors_Y;
  DMatrix<double> errors_X;
  DMatrix<double> errors_B;
  errors_Y.resize(n_batches, n_test_options * n_tests);
  errors_X.resize(n_batches, n_test_options * n_tests);
  errors_B.resize(n_batches, n_test_options * n_tests);

  // room for data and solutions
  DMatrix<double> B_clean;
  DMatrix<double> Y;
  DMatrix<double> X;
  DMatrix<double> Y_clean;
  DMatrix<double> X_clean;
  DMatrix<double> Y_hat;
  DMatrix<double> B_hat;
  DMatrix<double> X_hat;

  // loop
  for (unsigned int i : tests)
  {

    std::string test_directory = tests_directory + "test" + std::to_string(i) + "/";
    // std::string results_directory = test_directory + "results/";
    // if (!std::filesystem::exists(results_directory))
    //  std::filesystem::create_directory(results_directory);

    // load expected results data from .csv files
    CSVFile<double> bFile;
    bFile = reader.parseFile(test_directory + "B.csv");
    B_clean = bFile.toEigen();
    unsigned int n_nodes = B_clean.size();

    if (VERBOSE)
    {
      std::cout << std::endl;
      std::cout << "##########" << std::endl;
      std::cout << "# Test " << i << " #" << std::endl;
      std::cout << "##########" << std::endl;
      std::cout << std::endl;
    }

    for (unsigned int j = 1; j <= n_batches; ++j)
    {

      std::cout << "- Batch #" << j << ": ";

      // load data from .csv files
      CSVFile<double> yFile;
      yFile = reader.parseFile(test_directory + "Y_" + std::to_string(j) + ".csv");
      Y = yFile.toEigen();
      CSVFile<double> xFile;
      xFile = reader.parseFile(test_directory + "X_" + std::to_string(j) + ".csv");
      X = xFile.toEigen();

      // load expected results data from .csv files
      if (TEST)
      {
        yFile = reader.parseFile(test_directory + "Y_clean_" + std::to_string(j) + ".csv");
        Y_clean = yFile.toEigen();
        xFile = reader.parseFile(test_directory + "X_clean_" + std::to_string(j) + ".csv");
        X_clean = xFile.toEigen();
      }
      else
      {
        Y_clean = Y;
        X_clean = X;
      }
      unsigned int batch_size = Y_clean.rows();

      for (unsigned int t = 0; t < test_name_vect.size(); ++t)
      {

        std::string test_name = test_name_vect[t];
        bool smoothing_initialization = smoothing_initialization_vect[t];
        bool smoothing_regression = smoothing_regression_vect[t];
        bool full_functional = full_functional_vect[t];
        double lambda = lambda_vect[t];

        std::cout << test_name << " ";

        // set smoothing parameter
        model.setLambdaS(lambda);

        // disable smoothing for initialization and regression
        model.set_smoothing(smoothing_initialization, smoothing_regression);

        // full_functional: true -> harold's implementation, false -> correct implementation
        model.set_full_functional(full_functional);

        // set model data
        BlockFrame<double, int> df_data;
        df_data.insert(OBSERVATIONS_BLK, Y);
        df_data.insert(DESIGN_MATRIX_BLK, X);
        model.setData(df_data);

        // solve smoothing problem
        model.init();
        model.solve();

        //   **  export computed results  **   //

        // estimated quantities
        Y_hat = model.fitted();
        B_hat = model.B();
        X_hat = model.reconstructed_field();

        // compute errors
        errors_Y(j - 1, n_test_options * (i - 1) + t) = (Y_clean - Y_hat).squaredNorm() / batch_size;
        errors_X(j - 1, n_test_options * (i - 1) + t) = (X_clean - X_hat).squaredNorm() / (n_nodes * batch_size);
        if (TEST)
          errors_B(j - 1, n_test_options * (i - 1) + t) = (B_clean - B_hat).squaredNorm() / n_nodes;

        // output file
        std::ofstream outfile;

        // outfile.open(results_directory + "Y_hat_" + test_name + "_" + std::to_string(j) + ".csv");
        // outfile << Y_hat.format(Test_fPLSR_comparison::CSVFormat);
        // outfile.close();

        // outfile.open(results_directory + "B_hat_" + test_name + "_" + std::to_string(j) + ".csv");
        // outfile << B_hat.format(Test_fPLSR_comparison::CSVFormat);
        // outfile.close();

        // outfile.open(results_directory + "X_hat_" + test_name + "_" + std::to_string(j) + ".csv");
        // outfile << X_hat.format(Test_fPLSR_comparison::CSVFormat);
        // outfile.close();
      }
      std::cout << std::endl;
    }
  }

  // output file
  std::ofstream outfile;

  if (TEST)
    outfile.open(tests_directory + "errors_Y.csv");
  else
    outfile.open(tests_directory + "errors_Y_train.csv");
  outfile << errors_Y.format(Test_fPLSR_comparison::CSVFormat);
  outfile.close();

  if (TEST)
    outfile.open(tests_directory + "errors_X.csv");
  else
    outfile.open(tests_directory + "errors_X_train.csv");
  outfile << errors_X.format(Test_fPLSR_comparison::CSVFormat);
  outfile.close();

  if (TEST)
  {
    outfile.open(tests_directory + "errors_B.csv");
    outfile << errors_B.format(Test_fPLSR_comparison::CSVFormat);
    outfile.close();
  }
}