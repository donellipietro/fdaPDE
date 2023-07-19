## load all required modules

# regularizing PDEs
loadModule("Laplacian_2D_Order1", TRUE)
loadModule("Laplacian_3D_Order1", TRUE)
loadModule("Laplacian_3D_Order1", TRUE)
loadModule("ConstantCoefficients_2D_Order1", TRUE)
loadModule("SpaceVarying_2D_Order1", TRUE)

# fPCA
loadModule("FPCA_Laplacian_2D_GeoStatNodes", TRUE)
loadModule("FPCA_Laplacian_2D_GeoStatLocations", TRUE)
loadModule("FPCA_Laplacian_3D_GeoStatNodes", TRUE)
loadModule("FPCA_CS_Laplacian_2D_GeoStatNodes", TRUE)
loadModule("FPCA_CS_Laplacian_2D_GeoStatLocations", TRUE)
loadModule("FPCA_CS_Laplacian_3D_GeoStatNodes", TRUE)

# fSRPDE
loadModule("FSRPDE_Laplacian_2D_GeoStatNodes", TRUE)
loadModule("FSRPDE_Laplacian_2D_GeoStatLocations", TRUE)
