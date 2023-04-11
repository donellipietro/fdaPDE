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

  // BlockFrame<double, int> df_data{generate_2D_data(model, 50, 2)};

  // load data from .csv files
  CSVReader<double> reader{};
  CSVFile<double> yFile; // observation file
  CSVFile<double> xFile; // covariates file

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
    DMatrix<double> beta_hat{model.B()};

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

    file = reader.parseFile(test_directory + "beta_hat.csv");
    DMatrix<double> expected_beta_hat = file.toEigen();
    if (VERBOSE)
    {
      std::cout << "Expected:" << std::endl;
      std::cout << expected_beta_hat.topRows(5) << std::endl;
      std::cout << "Obtained:" << std::endl;
      std::cout << beta_hat.topRows(5) << std::endl;
      std::cout << "Error norm: " << (expected_beta_hat - beta_hat).lpNorm<Eigen::Infinity>() << std::endl;
      std::cout << "----------------" << std::endl;
    }
    EXPECT_TRUE(almost_equal(expected_beta_hat, beta_hat));

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

/* test 2
   domain:       unit square [1,1] x [1,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   BC:           no
   order FE:     1
 */
/*TEST(FPCA, Test2_Laplacian_GeostatisticalAtNodes_Separable_Monolithic) {
  // define time domain
  DVector<double> time_mesh;
  time_mesh.resize(10);
  std::size_t i = 0;
  for(double x = 0.5; x <= 0.95; x+=0.05, ++i) time_mesh[i] = x;

  // define domain and regularizing PDE
  MeshLoader<Mesh2D<>> domain("unit_square05");
  auto L = Laplacian();
  DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
  PDE problem(domain.mesh, L, u); // definition of regularizing PDE
  problem.init();

  // define statistical model
  // use optimal lambda to avoid possible numerical issues
  double lambdaS = 1e-2;
  double lambdaT = 1e-2;
  // defaults to monolithic solution
  FPCA<decltype(problem), fdaPDE::models::SpaceTimeSeparable,
       fdaPDE::models::Sampling::GeoStatMeshNodes, fdaPDE::models::fixed_lambda> model(problem, time_mesh);
  model.setLambdaS(lambdaS);
  model.setLambdaT(lambdaT);

  // load data from .csv files
  CSVReader<double> reader{};
  CSVFile<double> yFile; // observation file
  yFile = reader.parseFile("data/models/FPCA/2D_test2/y.csv");
  DMatrix<double> y = yFile.toEigen();

  // set model data
  BlockFrame<double, int> df;
  df.insert("y", DMatrix<double>(y.transpose()));
  model.setData(df);

  // std::vector<SVector<1>> lambdas;
  // for(double x = -6.0; x <= -2.0; x++) lambdas.push_back(SVector<1>(std::pow(10,x)));
  // model.setLambda(lambdas);

  // solve smoothing problem
  model.init();
  model.solve();

  //   **  test correctness of computed results  **

  // SpMatrix<double> expectedLoadings;
  // Eigen::loadMarket(expectedLoadings, "data/models/FPCA/2D_test1/loadings.mtx");
  // DMatrix<double> computedLoadings = model.loadings();
  // EXPECT_TRUE( almost_equal(DMatrix<double>(expectedLoadings), computedLoadings) );

  // SpMatrix<double> expectedScores;
  // Eigen::loadMarket(expectedScores,   "data/models/FPCA/2D_test1/scores.mtx");
  // DMatrix<double> computedScores = model.scores();
  // EXPECT_TRUE( almost_equal(DMatrix<double>(expectedScores), computedScores) );

}*/

/* test 2
   domain:       unit square [1,1] x [1,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   BC:           no
   order FE:     1
 */
/*TEST(FPCA, Test2_Laplacian_GeostatisticalAtNodes_Separable_Monolithic) {
  // define time domain
  DVector<double> time_mesh;
  time_mesh.resize(11);
  std::size_t i = 0;
  for(double x = 0; x <= 0.5; x+=0.05, ++i) time_mesh[i] = x;

  // define domain and regularizing PDE
  MeshLoader<Mesh2D<>> domain("unit_square_coarse");
  auto L = Laplacian();
  DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements()*3, 1);
  PDE problem(domain.mesh, L, u); // definition of regularizing PDE

  // define statistical model
  // use optimal lambda to avoid possible numerical issues
  double lambdaS = 1e-4;
  double lambdaT = 1e-4;
  // defaults to monolithic solution
  FPCA<decltype(problem), fdaPDE::models::SpaceTimeSeparable,
       fdaPDE::models::Sampling::GeoStatMeshNodes, fdaPDE::models::gcv_lambda_selection> model;
  model.setPDE(problem);
  model.setTimeDomain(time_mesh);
  model.setLambdaS(lambdaS);
  model.setLambdaT(lambdaT);

  // load data from .csv files
  CSVReader<double> reader{};
  CSVFile<double> yFile; // observation file
  yFile = reader.parseFile("data/models/FPCA/2D_test3/y.csv");
  DMatrix<double> y = yFile.toEigen().leftCols(11*441);

  // set model data
  BlockFrame<double, int> df;
  df.insert("y", DMatrix<double>(y.transpose()));
  model.setData(df);

  std::vector<SVector<2>> lambdas;
  for(double x = -4.0; x <= -2.0; x+=0.5) {
    for(double y = -4.0; y <= -2.0; y+=0.5) {
      lambdas.push_back(SVector<2>(std::pow(10,x), std::pow(10,y)));
    }
  }
  model.setLambda(lambdas);

  // solve smoothing problem
  model.init();
  model.solve();

  //std::cout << model.loadings() << std::endl;

  //   **  test correctness of computed results  **

  // SpMatrix<double> expectedLoadings;
  // Eigen::loadMarket(expectedLoadings, "data/models/FPCA/2D_test1/loadings.mtx");
  // DMatrix<double> computedLoadings = model.loadings();
  // EXPECT_TRUE( almost_equal(DMatrix<double>(expectedLoadings), computedLoadings) );

  // SpMatrix<double> expectedScores;
  // Eigen::loadMarket(expectedScores,   "data/models/FPCA/2D_test1/scores.mtx");
  // DMatrix<double> computedScores = model.scores();
  // EXPECT_TRUE( almost_equal(DMatrix<double>(expectedScores), computedScores) );

}*/
