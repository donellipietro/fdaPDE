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
    MeshLoader<Mesh2D<>> domain("unit_square");
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

    //  **  export results  ** //

    // std::string results_directory = test_directory + "results/";

    // if (!std::filesystem::exists(results_directory))
    //     std::filesystem::create_directory(results_directory);

    // std::ofstream nodesfile;
    // nodesfile.open(results_directory + "f.csv");
    // DMatrix<double> computedF = model.f();
    // nodesfile << computedF.format(Test_FRPDE::CSVFormat);
    // nodesfile.close();
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

    // //  **  export results  ** //

    // std::string results_directory = test_directory + "results/";

    // if (!std::filesystem::exists(results_directory))
    //     std::filesystem::create_directory(results_directory);

    // std::ofstream nodesfile;
    // nodesfile.open(results_directory + "f.csv");
    // DMatrix<double> computedF = model.f();
    // nodesfile << computedF.format(Test_FRPDE::CSVFormat);
    // nodesfile.close();
}
