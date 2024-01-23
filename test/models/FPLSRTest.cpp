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

#include <string>
#include <iomanip>

void save_mtx(const DMatrix<double> &matrix, const std::string &file_path)
{
    std::ofstream file(file_path);

    // Set the output format to scientific notation with 18 digits
    file << std::scientific << std::setprecision(18);

    file << "%%MatrixMarket matrix coordinate real general\n";
    file << "%\n";
    file << matrix.rows() << " " << matrix.cols() << " " << matrix.size() << "\n";

    for (int i = 0; i < matrix.rows(); i++)
    {
        for (int j = 0; j < matrix.cols(); j++)
        {
            double value = matrix(i, j);
            if (value != 0)
            {
                file << i + 1 << " " << j + 1 << " " << value << "\n";
            }
        }
    }

    file.close();
}

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

}

/* test 1:
   Calibration: Off
   approach:     NIPALS
   domain:       unit square [0,1] x [0,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   BC:           no
   order FE:     1
 */
TEST(FPLSR, Test1_fPLS_samplingAtLocations_calibrationOff)
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
    std::string test_directory = "data/models/FPLSR/2D_test1/";
    std::string results_directory = "data/models/FPLSR/2D_test1/results_calibrationOff/";
    bool VERBOSE = true;

    // options
    std::vector<SVector<1>> lambdas{SVector<1>{10.0}};
    bool full_functional = false;

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
    model.setLambda(lambdas);                          // covariance maximization
    model.set_smoothing_initialization(true, lambdas); // centering
    model.set_smoothing_regression(true, lambdas);     // regression
    model.set_full_functional(full_functional);

    // set model data
    BlockFrame<double, int> df_data;
    df_data.insert(OBSERVATIONS_BLK, Y);
    df_data.insert(DESIGN_MATRIX_BLK, X);
    model.setData(df_data);

    std::cout << "cioa" << std::endl;

    // solve smoothing problem
    model.init();

    std::cout << "cioa" << std::endl;

    model.solve();

    std::cout << "cioa_ext" << std::endl;

    //   **  test correctness of computed results  **   //

    save_mtx(model.W(), test_directory + "W_hat.csv");
    save_mtx(model.V(), test_directory + "V_hat.csv");
    save_mtx(model.T(), test_directory + "T_hat.csv");
    save_mtx(model.C(), test_directory + "C_hat.csv");
    save_mtx(model.D(), test_directory + "D_hat.csv");
    save_mtx(model.fitted(), test_directory + "Y_hat.csv");
    save_mtx(model.Y_mean(), test_directory + "Y_mean.csv");
    save_mtx(model.reconstructed_field(), test_directory + "X_hat.csv");
    save_mtx(model.X_mean(), test_directory + "X_mean.csv");
    save_mtx(model.B(), test_directory + "B_hat.csv");

    std::cout << "lambda_initialization: ";
    for (auto lambda : model.get_lambda_initialization())
        std::cout << lambda << " ";
    std::cout << std::endl;

    std::cout << "lambda_directions: ";
    for (auto lambda : model.get_lambda_directions())
        std::cout << lambda << " ";
    std::cout << std::endl;

    std::cout << "lambda_regression: ";
    for (auto lambda : model.get_lambda_regression())
        std::cout << lambda << " ";
    std::cout << std::endl;
}

TEST(FPLSR, Test2_fPLS_samplingAtLocations_calibrationGcv)
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
    std::string test_directory = "data/models/FPLSR/2D_test1/";
    std::string results_directory = "data/models/FPLSR/2D_test1/results_calibrationGcv/";
    bool VERBOSE = false;

    // options
    std::vector<SVector<1>> lambdas;
    for (double x = -4; x <= 0; x += 1)
        lambdas.push_back(SVector<1>(std::pow(10, x)));
    bool full_functional = false;

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
    model.setLambda(lambdas);                          // covariance maximization
    model.set_smoothing_initialization(true, lambdas); // centering
    model.set_smoothing_regression(true, lambdas);     // regression
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

    save_mtx(model.W(), results_directory + "W_hat.csv");
    save_mtx(model.V(), results_directory + "V_hat.csv");
    save_mtx(model.T(), results_directory + "T_hat.csv");
    save_mtx(model.C(), results_directory + "C_hat.csv");
    save_mtx(model.D(), results_directory + "D_hat.csv");
    save_mtx(model.fitted(), results_directory + "Y_hat.csv");
    save_mtx(model.Y_mean(), results_directory + "Y_mean.csv");
    save_mtx(model.reconstructed_field(), results_directory + "X_hat.csv");
    save_mtx(model.X_mean(), results_directory + "X_mean.csv");
    save_mtx(model.B(), results_directory + "B_hat.csv");

    std::cout << "lambda_initialization: ";
    for (auto lambda : model.get_lambda_initialization())
        std::cout << lambda << " ";
    std::cout << std::endl;

    std::cout << "lambda_directions: ";
    for (auto lambda : model.get_lambda_directions())
        std::cout << lambda << " ";
    std::cout << std::endl;

    std::cout << "lambda_regression: ";
    for (auto lambda : model.get_lambda_regression())
        std::cout << lambda << " ";
    std::cout << std::endl;
}
