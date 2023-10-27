#include <cstddef>
#include <gtest/gtest.h> // testing framework
#include <unsupported/Eigen/SparseExtra>

#include "../fdaPDE/core/utils/Symbols.h"
#include "../fdaPDE/core/utils/IO/CSVReader.h"
#include "../fdaPDE/core/FEM/PDE.h"
using fdaPDE::core::FEM::PDE;
#include "core/MESH/Mesh.h"
#include "../fdaPDE/models/functional/fPCA_CS.h"
using fdaPDE::models::FPCA_CS;
#include "../fdaPDE/models/SamplingDesign.h"
#include "../../fdaPDE/models/ModelTraits.h"

#include "../utils/MeshLoader.h"
using fdaPDE::testing::MeshLoader;
#include "../utils/Constants.h"
using fdaPDE::testing::DOUBLE_TOLERANCE;
#include "../utils/Utils.h"
using fdaPDE::testing::almost_equal;

/* test 1
   domain:       unit square [1,1] x [1,1]
   sampling:     locations = nodes
   penalization: simple laplacian
   BC:           no
   order FE:     1
   missing data: no
 */
TEST(FPCA_CS, Test1_Laplacian_GeostatisticalAtNodes_Fixed)
{

    bool VERBOSE = false;

    // define domain and regularizing PDE
    MeshLoader<Mesh2D<>> domain("unit_square");
    auto L = Laplacian();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.elements() * 3, 1);
    PDE problem(domain.mesh, L, u); // definition of regularizing PDE

    // define statistical model
    double lambda = 1e-2;

    FPCA_CS<decltype(problem), fdaPDE::models::SpaceOnly, fdaPDE::models::GeoStatMeshNodes,
            fdaPDE::models::fixed_lambda>
        model(problem);
    model.setLambdaS(lambda);
    model.set_verbose(VERBOSE);
    model.set_mass_lumping(true);
    model.set_iterative(false);

    // load data from .csv files
    CSVReader<double> reader{};
    CSVFile<double> yFile; // observation file
    yFile = reader.parseFile("data/models/FPCA/2D_test1/y.csv");
    DMatrix<double> y = yFile.toEigen();

    // set model data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.setData(df);

    // solve smoothing problem
    model.init();
    model.solve();

    //   **  test correctness of computed results  **

    // loadings vector
    SpMatrix<double> expectedLoadings;
    Eigen::loadMarket(expectedLoadings, "data/models/FPCA/2D_test1/loadings_CS.mtx");
    DMatrix<double> computedLoadings = model.loadings();
    if (VERBOSE)
    {
        std::cout << "\nExpected loadings:" << std::endl;
        std::cout << expectedLoadings.topRows(5) << std::endl;
        std::cout << "Obtained loadings:" << std::endl;
        std::cout << computedLoadings.topRows(5) << std::endl;
    }
    EXPECT_TRUE(almost_equal(DMatrix<double>(expectedLoadings), computedLoadings));

    // scores vector
    SpMatrix<double> expectedScores;
    Eigen::loadMarket(expectedScores, "data/models/FPCA/2D_test1/scores_CS.mtx");
    DMatrix<double> computedScores = model.scores();
    if (VERBOSE)
    {
        std::cout << "\nExpected scores:" << std::endl;
        std::cout << expectedScores.topRows(5) << std::endl;
        std::cout << "Obtained scores:" << std::endl;
        std::cout << computedScores.topRows(5) << std::endl;
    }
    EXPECT_TRUE(almost_equal(DMatrix<double>(expectedScores), computedScores));
}
