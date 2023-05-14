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

const static Eigen::IOFormat CSVFormat2(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "\n");

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
  bool VERBOSE = false;
  std::vector<unsigned int> tests{1, 2, 3, 4, 5, 6};
  unsigned int n_batches = 20;

  // reader
  CSVReader<double> reader{};

  // Tests options
  std::vector<std::string> test_name_vect = {"hcpp", "sr", "sri"};
  std::vector<bool> smoothing_initialization_vect = {false, false, true};
  std::vector<bool> smoothing_regression_vect = {false, true, true};
  std::vector<bool> full_functional_vect = {true, false, false};

  for (unsigned int t = 0; t < test_name_vect.size(); ++t)
  {

    std::string test_name = test_name_vect[t];
    bool smoothing_initialization = smoothing_initialization_vect[t];
    bool smoothing_regression = smoothing_regression_vect[t];
    bool full_functional = full_functional_vect[t];

    for (unsigned int i : tests)
    {

      if (VERBOSE)
      {
        std::cout << "##########" << std::endl;
        std::cout << "# Test " << i << " #" << std::endl;
        std::cout << "##########" << std::endl;
      }

      std::string test_directory = tests_directory + "test" + std::to_string(i) + "/";
      std::string results_directory = test_directory + "results/";
      if (!std::filesystem::exists(results_directory))
        std::filesystem::create_directory(results_directory);

      for (unsigned int j = 1; j <= n_batches; ++j)
      {

        // load data from .csv files
        CSVFile<double> yFile;
        yFile = reader.parseFile(test_directory + "Y_" + std::to_string(j) + ".csv");
        DMatrix<double> Y = yFile.toEigen();
        CSVFile<double> xFile;
        xFile = reader.parseFile(test_directory + "X_" + std::to_string(j) + ".csv");
        DMatrix<double> X = xFile.toEigen();

        // set smoothing parameter
        double lambda = 10;
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

        // output file
        std::ofstream outfile;

        // Results
        DMatrix<double> Y_hat{model.fitted()};
        DMatrix<double> B_hat{model.B()};
        DMatrix<double> X_hat{model.reconstructed_field()};

        outfile.open(results_directory + "Y_hat_" + test_name + "_" + std::to_string(j) + ".csv");
        outfile << Y_hat.format(CSVFormat2);
        outfile.close();

        outfile.open(results_directory + "B_hat_" + test_name + "_" + std::to_string(j) + ".csv");
        outfile << B_hat.format(CSVFormat2);
        outfile.close();

        outfile.open(results_directory + "X_hat_" + test_name + "_" + std::to_string(j) + ".csv");
        outfile << X_hat.format(CSVFormat2);
        outfile.close();
      }
    }
  }
}