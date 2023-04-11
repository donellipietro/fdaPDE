#include <gtest/gtest.h> // testing framework
// include eigen now to avoid possible linking errors
#include <Eigen/Dense>
#include <Eigen/Sparse>

// fields test suites
#include "core/ScalarFieldTest.cpp"
#include "core/VectorFieldTest.cpp"
// MESH test suites
#include "core/MeshTest.cpp"
#include "core/ElementTest.cpp"
#include "core/SearchEngineTest.cpp"
// NLA test suites
#include "core/VectorSpaceTest.cpp"
#include "core/KroneckerProductTest.cpp"
// FEM test suites
#include "core/LagrangianBasisTest.cpp"
#include "core/IntegratorTest.cpp"
#include "core/BilinearFormsTest.cpp"
#include "core/PDESolutionsTest.cpp"
// space-time test suites
#include "core/SplineTest.cpp"
#include "models/SpaceTimeTest.cpp"
// regression module test suites
#include "models/SRPDETest.cpp"
#include "models/STRPDETest.cpp"
#include "models/GSRPDETest.cpp"
// #include "models/FPCATest.cpp"
#include "models/FPLSRTest.cpp"
// GCV test suites
#include "calibration/GCVTest.cpp"
#include "calibration/GCVNewtonTest.cpp"

int main(int argc, char **argv)
{
  // start testing
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
