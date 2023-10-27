#include <cstddef>
#include <gtest/gtest.h> // testing framework
#include <unsupported/Eigen/SparseExtra>

#include "../fdaPDE/core/utils/Symbols.h"
#include "../fdaPDE/core/utils/IO/CSVReader.h"
#include "../fdaPDE/core/FEM/PDE.h"
using fdaPDE::core::FEM::PDE;
#include "../fdaPDE/core/FEM/operators/SpaceVaryingFunctors.h"
using fdaPDE::core::FEM::SpaceVaryingAdvection;
using fdaPDE::core::FEM::SpaceVaryingDiffusion;
#include "core/MESH/Mesh.h"
#include "../fdaPDE/models/functional/FRPDE.h"
using fdaPDE::models::FRPDE;
#include "../fdaPDE/models/SamplingDesign.h"

#include "../utils/MeshLoader.h"
using fdaPDE::testing::MeshLoader;
#include "../utils/Constants.h"
using fdaPDE::testing::DOUBLE_TOLERANCE;
#include "../utils/Utils.h"
using fdaPDE::testing::almost_equal;

#include <string>
#include <fstream>
#include <filesystem>

namespace Test_FRPDE
{
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "\n");
}

/* test 1
   domain:       unit square [0,1] x [0,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   covariates:   no
   BC:           no
   order FE:     1
 */
TEST(FRPDE, Test1_Laplacian_NonParametric_GeostatisticalAtNodes)
{
    bool VERBOSE = false;

    // define domain and regularizing PDE
    MeshLoader<Mesh2D<>>
        domain("unit_square");
    auto L = Laplacian();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements() * 3, 1);
    PDE problem(domain.mesh, L, u); // definition of regularizing PDE

    // define statistical model
    FRPDE<decltype(problem), fdaPDE::models::GeoStatMeshNodes> model(problem);

    // set lambda
    double lambda = std::pow(0.1, 4);
    model.setLambdaS(lambda);
    model.set_verbose(VERBOSE);

    // load data from .csv files
    std::string test_directory = "data/models/FRPDE/2D_test/";
    CSVReader<double> reader{};
    CSVFile<double> XFile; // observation file
    XFile = reader.parseFile(test_directory + "X.csv");
    DMatrix<double> X = XFile.toEigen();
    CSVFile<double> bFile; // observation file
    bFile = reader.parseFile(test_directory + "b.csv");
    DMatrix<double> b = bFile.toEigen();

    // set model data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, X);
    df.insert(DESIGN_MATRIX_BLK, b);
    model.setData(df);
    // alternative
    // model.setData(X);

    // solve smoothing problem
    model.init();
    model.solve();

    //   **  test correctness of computed results  **

    SpMatrix<double> expectedF;
    Eigen::loadMarket(expectedF, "data/models/FRPDE/2D_test/f.mtx");
    DMatrix<double> computedF = model.f();
    if (VERBOSE)
    {
        std::cout << "\nExpected f:" << std::endl;
        std::cout << expectedF.leftCols(5) << std::endl;
        std::cout << "Obtained f:" << std::endl;
        std::cout << computedF.leftCols(5) << std::endl;
    }
    EXPECT_TRUE(almost_equal(DMatrix<double>(expectedF), computedF));

    SpMatrix<double> expectedFitted;
    Eigen::loadMarket(expectedFitted, "data/models/FRPDE/2D_test/fitted.mtx");
    DMatrix<double> computedFitted = model.fitted();
    if (VERBOSE)
    {
        std::cout << "\nExpected fitted:" << std::endl;
        std::cout << expectedFitted.leftCols(5) << std::endl;
        std::cout << "Obtained fitted:" << std::endl;
        std::cout << computedFitted.leftCols(5) << std::endl;
    }
    EXPECT_TRUE(almost_equal(DMatrix<double>(expectedFitted), computedFitted));
}

/* test 2
   domain:       unit square [0,1] x [0,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   covariates:   no
   BC:           no
   order FE:     1
   lambda selevtion: GCV
 */
TEST(FRPDE, Test1_Laplacian_NonParametric_GeostatisticalAtNodes_GCV)
{
    bool VERBOSE = false;

    // define domain and regularizing PDE
    MeshLoader<Mesh2D<>>
        domain("unit_square");
    auto L = Laplacian();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements() * 3, 1);
    PDE problem(domain.mesh, L, u); // definition of regularizing PDE

    // define statistical model
    FRPDE<decltype(problem), fdaPDE::models::GeoStatMeshNodes> model(problem);

    // set lambda
    double lambda = std::pow(0.1, 4);
    model.setLambdaS(lambda);
    model.set_verbose(VERBOSE);

    // load data from .csv files
    std::string test_directory = "data/models/FRPDE/2D_test/";
    CSVReader<double> reader{};
    CSVFile<double> XFile; // observation file
    XFile = reader.parseFile(test_directory + "X.csv");
    DMatrix<double> X = XFile.toEigen();
    CSVFile<double> bFile; // observation file
    bFile = reader.parseFile(test_directory + "b.csv");
    DMatrix<double> b = bFile.toEigen();

    // lambdas
    std::vector<SVector<1>> lambdas;
    for (double x = -6.0; x <= 0.0; x += 0.5)
        lambdas.push_back(SVector<1>(std::pow(10, x)));

    model.tune_and_compute(X, b, lambdas);

    //   **  test correctness of computed results  **

    SpMatrix<double> expectedF;
    Eigen::loadMarket(expectedF, "data/models/FRPDE/2D_test/f_GCV.mtx");
    DMatrix<double> computedF = model.f();
    if (VERBOSE)
    {
        std::cout << "\nExpected f:" << std::endl;
        std::cout << expectedF.leftCols(5) << std::endl;
        std::cout << "Obtained f:" << std::endl;
        std::cout << computedF.leftCols(5) << std::endl;
    }
    EXPECT_TRUE(almost_equal(DMatrix<double>(expectedF), computedF));

    SpMatrix<double> expectedFitted;
    Eigen::loadMarket(expectedFitted, "data/models/FRPDE/2D_test/fitted_GCV.mtx");
    DMatrix<double> computedFitted = model.fitted();
    if (VERBOSE)
    {
        std::cout << "\nExpected fitted:" << std::endl;
        std::cout << expectedFitted.leftCols(5) << std::endl;
        std::cout << "Obtained fitted:" << std::endl;
        std::cout << computedFitted.leftCols(5) << std::endl;
    }
    EXPECT_TRUE(almost_equal(DMatrix<double>(expectedFitted), computedFitted));
}

TEST(FRPDE, Test2_Surface_domain_at_locations)
{
    bool VERBOSE = false;

    std::string test_directory = "data/models/FRPDE/2.5D_test/";
    CSVReader<double> reader{};

    // define domain and regularizing PDE
    MeshLoader<SurfaceMesh<>> domain("cylinder");
    auto L = Laplacian();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements() * 3, 1);
    PDE problem(domain.mesh, L, u); // definition of regularizing PDE

    // define statistical model
    FRPDE<decltype(problem), fdaPDE::models::GeoStatLocations> model(problem);

    // load locations from .csv files
    CSVFile<double> locsFile;
    locsFile = reader.parseFile(test_directory + "locations.csv");
    DMatrix<double> locs = locsFile.toEigen();
    model.set_spatial_locations(locs);

    // set the lamnda
    double lambda = std::pow(0.1, 2);
    model.setLambdaS(lambda);
    model.set_verbose(VERBOSE);

    // load data from .csv files
    CSVFile<double> XFile; // observation file
    XFile = reader.parseFile(test_directory + "X.csv");
    DMatrix<double> X = XFile.toEigen();

    // set model data
    BlockFrame<double, int> df;
    model.setData(X);

    // solve smoothing problem
    model.init();
    model.solve();

    //   **  test correctness of computed results  **

    SpMatrix<double> expectedF;
    Eigen::loadMarket(expectedF, "data/models/FRPDE/2.5D_test/f.mtx");
    DMatrix<double> computedF = model.f();
    if (VERBOSE)
    {
        std::cout << "\nExpected f:" << std::endl;
        std::cout << expectedF.leftCols(5) << std::endl;
        std::cout << "Obtained f:" << std::endl;
        std::cout << computedF.leftCols(5) << std::endl;
    }
    EXPECT_TRUE(almost_equal(DMatrix<double>(expectedF), computedF));

    SpMatrix<double> expectedFitted;
    Eigen::loadMarket(expectedFitted, "data/models/FRPDE/2.5D_test/fitted.mtx");
    DMatrix<double> computedFitted = model.fitted();
    if (VERBOSE)
    {
        std::cout << "\nExpected fitted:" << std::endl;
        std::cout << expectedFitted.leftCols(5) << std::endl;
        std::cout << "Obtained fitted:" << std::endl;
        std::cout << computedFitted.leftCols(5) << std::endl;
    }
    EXPECT_TRUE(almost_equal(DMatrix<double>(expectedFitted), computedFitted));
}
