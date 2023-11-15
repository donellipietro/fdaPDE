## load all required modules

# regularizing PDEs
loadModule("Laplacian_2D_Order1", TRUE)
loadModule("Laplacian_Surface_Order1", TRUE)
loadModule("Laplacian_3D_Order1", TRUE)
loadModule("ConstantCoefficients_2D_Order1", TRUE)
loadModule("SpaceVarying_2D_Order1", TRUE)

# fPCA
loadModule("FPCA_Laplacian_2D_GeoStatNodes_fixed", TRUE)
loadModule("FPCA_Laplacian_2D_GeoStatLocations_fixed", TRUE)
loadModule("FPCA_Laplacian_3D_GeoStatNodes_fixed", TRUE)
loadModule("FPCA_Laplacian_2D_GeoStatNodes_GCV", TRUE)
loadModule("FPCA_Laplacian_2D_GeoStatLocations_GCV", TRUE)
loadModule("FPCA_Laplacian_3D_GeoStatNodes_GCV", TRUE)
loadModule("FPCA_CS_Laplacian_2D_GeoStatNodes", TRUE)
loadModule("FPCA_CS_Laplacian_2D_GeoStatLocations", TRUE)
loadModule("FPCA_CS_Laplacian_3D_GeoStatNodes", TRUE)


# FRPDE
loadModule("FRPDE_Laplacian_2D_GeoStatNodes", TRUE)
loadModule("FRPDE_Laplacian_2D_GeoStatLocations", TRUE)
loadModule("FRPDE_Laplacian_Surface_GeoStatNodes", TRUE)
loadModule("FRPDE_Laplacian_Surface_GeoStatLocations", TRUE)

# FPLSR
loadModule("FPLSR_Laplacian_2D_GeoStatNodes", TRUE)
loadModule("FPLSR_Laplacian_2D_GeoStatLocations", TRUE)
