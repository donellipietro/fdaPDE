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

namespace Test_fPLSR_SIMPLS
{

    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "\n");

    void compareAndExportResults(const std::string &test_directory,
                                 const std::string &results_directory,
                                 const bool VERBOSE,
                                 const DMatrix<double> &Y_hat,
                                 const DMatrix<double> &X_hat,
                                 const DMatrix<double> &B_hat,
                                 std::vector<double> &errors_Y,
                                 std::vector<double> &errors_X,
                                 std::vector<double> &errors_B)
    {

        // reader
        CSVReader<double> reader{};

        // output file
        std::ofstream outfile;

        // dimensions
        const unsigned int N = X_hat.rows();
        const unsigned int S = X_hat.cols();

        CSVFile<double> file; // covariates file

        // Y
        file = reader.parseFile(test_directory + "Y_clean.csv");
        DMatrix<double> Y_clean = file.toEigen();
        errors_Y.push_back((Y_clean - Y_hat).squaredNorm() / N);
        if (VERBOSE)
        {
            std::cout << std::endl;
            std::cout << "||||||| Y ||||||||:" << std::endl;
            std::cout << "Clean:" << std::endl;
            std::cout << Y_clean.topRows(5) << std::endl;
            std::cout << "Prediction:" << std::endl;
            std::cout << Y_hat.topRows(5) << std::endl;
            std::cout << "Error norm: " << errors_Y.back() << std::endl;
            std::cout << "----------------" << std::endl;
            std::cout << std::endl;
        }
        outfile.open(results_directory + "Y_hat.csv");
        outfile << Y_hat.format(Test_fPLSR_SIMPLS::CSVFormat);
        outfile.close();

        // X
        file = reader.parseFile(test_directory + "X_clean.csv");
        DMatrix<double> X_clean = file.toEigen();
        errors_X.push_back((X_clean - X_hat).squaredNorm() / (N * S));
        if (VERBOSE)
        {
            std::cout << std::endl;
            std::cout << "||||||| X ||||||||:" << std::endl;
            std::cout << "Clean:" << std::endl;
            std::cout << X_clean.block(0, 0, 5, 5) << std::endl;
            std::cout << "Prediction:" << std::endl;
            std::cout << X_hat.block(0, 0, 5, 5) << std::endl;
            std::cout << "Error norm: " << errors_X.back() << std::endl;
            std::cout << "----------------" << std::endl;
            std::cout << std::endl;
        }
        outfile.open(results_directory + "X_hat.csv");
        outfile << X_hat.format(Test_fPLSR_SIMPLS::CSVFormat);
        outfile.close();

        // B
        file = reader.parseFile(test_directory + "B.csv");
        DMatrix<double> B = file.toEigen();
        errors_B.push_back((B - B_hat).squaredNorm() / (S));
        if (VERBOSE)
        {
            std::cout << std::endl;
            std::cout << "||||||| B ||||||||:" << std::endl;
            std::cout << "Expected:" << std::endl;
            std::cout << B.topRows(5) << std::endl;
            std::cout << "Obtained version:" << std::endl;
            std::cout << B_hat.topRows(5) << std::endl;
            std::cout << "Error norm: " << errors_B.back() << std::endl;
            std::cout << "----------------" << std::endl;
            std::cout << std::endl;
        }
        outfile.open(results_directory + "B_hat.csv");
        outfile << B_hat.format(Test_fPLSR_SIMPLS::CSVFormat);
        outfile.close();
    }

    void exportErrors(const std::string &tests_directory,
                      const std::vector<unsigned int> &tests,
                      const bool VERBOSE,
                      const std::vector<double> &errors_Y,
                      const std::vector<double> &errors_X,
                      const std::vector<double> &errors_B,
                      const std::string &subname = "")
    {
        std::ofstream results(tests_directory + "errors" + subname + ".csv");
        if (VERBOSE)
        {
            std::cout << std::endl;
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
        if (VERBOSE)
            std::cout << std::endl;

        results.close();
    }

}

/* test 1:
   domain:       unit square [0,1] x [0,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   BC:           no
   order FE:     1
 */
TEST(FPLSR_SIMPLS, Test1_Laplacian_GeostatisticalAtNodes)
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

    // tests
    std::string tests_directory = "data/models/FPLSR_SIMPLS/2D_test1/";

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

        Test_fPLSR_SIMPLS::compareAndExportResults(test_directory,
                                                   results_directory,
                                                   VERBOSE,
                                                   Y_hat,
                                                   X_hat,
                                                   B_hat,
                                                   errors_Y,
                                                   errors_X,
                                                   errors_B);
    }

    Test_fPLSR_SIMPLS::exportErrors(tests_directory,
                                    tests,
                                    VERBOSE,
                                    errors_Y,
                                    errors_X,
                                    errors_B);
}

/* test 2:
   domain:       unit square [0,1] x [0,1]
   sampling:     locations != nodes, #locations == #nodes
   penalization: simple laplacian
   BC:           no
   order FE:     1
 */
TEST(FPLSR_SIMPLS, Test2_Laplacian_AtLocations)
{

    // define domain and regularizing PDE
    MeshLoader<Mesh2D<>> domain("unit_square");
    auto L = Laplacian();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements() * 3, 1);
    PDE problem(domain.mesh, L, u); // definition of regularizing PDE

    // define statistical model
    FPLSR_SIMPLS<decltype(problem), SpaceOnly, fdaPDE::models::GeoStatLocations,
                 fdaPDE::models::fixed_lambda>
        model(problem);

    // tests
    std::string tests_directory = "data/models/FPLSR_SIMPLS/2D_test2/";

    bool VERBOSE = false;
    std::vector<unsigned int> tests{1, 2, 3, 4, 5, 6};

    // reader
    CSVReader<double> reader{};

    // output file
    std::ofstream outfile;

    // for (unsigned int x = 1; x <= 6; ++x)
    // {

    // room to store the errors
    std::vector<double> errors_Y;
    std::vector<double> errors_X;
    std::vector<double> errors_B;
    errors_Y.reserve(tests.size());
    errors_X.reserve(tests.size());
    errors_B.reserve(tests.size());

    // double lambda_smoothing = pow(10, -static_cast<double>(x));
    double lambda_smoothing = 1e-3;

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
        std::string results_directory = test_directory + "results" + /* "_lambda1e-" + std::to_string(x) + */ "/";
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

        // set smoothing in initialization and regression
        model.set_smoothing(true, true, lambda_smoothing, lambda_smoothing);

        // set number of latent components
        // model.set_H(3);

        // load data from .csv files
        CSVFile<double>
            yFile; // observation file
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

        Test_fPLSR_SIMPLS::compareAndExportResults(test_directory,
                                                   results_directory,
                                                   VERBOSE,
                                                   Y_hat,
                                                   X_hat,
                                                   B_hat,
                                                   errors_Y,
                                                   errors_X,
                                                   errors_B);
    }

    Test_fPLSR_SIMPLS::exportErrors(tests_directory,
                                    tests,
                                    VERBOSE,
                                    errors_Y,
                                    errors_X,
                                    errors_B /*, "_lambda1e-" + std::to_string(x)*/);
    //}
}

/* test 3:
   domain:       unit square [0,1] x [0,1]
   sampling:     locations != nodes, #locations < #nodes
   penalization: simple laplacian
   BC:           no
   order FE:     1
 */
TEST(FPLSR_SIMPLS, Test3_Laplacian_AtLocations)
{

    // define domain and regularizing PDE
    MeshLoader<Mesh2D<>> domain("unit_square");
    auto L = Laplacian();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements() * 3, 1);
    PDE problem(domain.mesh, L, u); // definition of regularizing PDE

    // define statistical model
    FPLSR_SIMPLS<decltype(problem), SpaceOnly, fdaPDE::models::GeoStatLocations,
                 fdaPDE::models::fixed_lambda>
        model(problem);

    // tests
    std::string tests_directory = "data/models/FPLSR_SIMPLS/2D_test3/";

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

        Test_fPLSR_SIMPLS::compareAndExportResults(test_directory,
                                                   results_directory,
                                                   VERBOSE,
                                                   Y_hat,
                                                   X_hat,
                                                   B_hat,
                                                   errors_Y,
                                                   errors_X,
                                                   errors_B);
    }

    Test_fPLSR_SIMPLS::exportErrors(tests_directory,
                                    tests,
                                    VERBOSE,
                                    errors_Y,
                                    errors_X,
                                    errors_B);
}

/* test 4:
   domain:       unit square [0,1] x [0,1]
   sampling:     locations != nodes, #locations << #nodes in a equispaced subgrid
   penalization: simple laplacian
   BC:           no
   order FE:     1
 */
TEST(FPLSR_SIMPLS, Test4_Laplacian_AtLocations)
{

    // define domain and regularizing PDE
    MeshLoader<Mesh2D<>> domain("unit_square");
    auto L = Laplacian();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements() * 3, 1);
    PDE problem(domain.mesh, L, u); // definition of regularizing PDE

    // define statistical model
    FPLSR_SIMPLS<decltype(problem), SpaceOnly, fdaPDE::models::GeoStatLocations,
                 fdaPDE::models::fixed_lambda>
        model(problem);

    // tests
    std::string tests_directory = "data/models/FPLSR_SIMPLS/2D_test4/";

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

        Test_fPLSR_SIMPLS::compareAndExportResults(test_directory,
                                                   results_directory,
                                                   VERBOSE,
                                                   Y_hat,
                                                   X_hat,
                                                   B_hat,
                                                   errors_Y,
                                                   errors_X,
                                                   errors_B);
    }

    Test_fPLSR_SIMPLS::exportErrors(tests_directory,
                                    tests,
                                    VERBOSE,
                                    errors_Y,
                                    errors_X,
                                    errors_B);
}