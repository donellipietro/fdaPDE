smooth = function(domain, observations,
                  covariates = NULL, locations = NULL, incidence_matrix = NULL,
                  pde.parameters = NULL) {

    ## create Rcpp module
    pde = new(PDE.2D, domain = domain)

    # perform proper preprocessing and sanity checks before call C++ layer
    
    pde.parameters.eval = NULL
    if(!is.null(pde.parameters)){
        ## pde.parameters are passed as R functions, the C++ layer accepts space varying PDEs
        ## where the PDE parameters are given as matrices. The C++ layer expects each matrix
        ## element to represent the evaluation of the parameter at each quadrature node

        ## requrest quadrature nodes from the C++ layer
        quadrature_nodes = pde$get_quadrature_nodes()

        ## evaluate each function at the quadrature nodes
        if("K" %in% names(pde.parameters) &&
           !is.null(pde.parameters[["K"]])) ## diffusion
            pde.parameters.eval$K = apply(quadrature_nodes, 1, pde.parameters$K)
        if("b" %in% names(pde.parameters) &&
           !is.null(pde.parameters[["b"]])) ## advection
            pde.parameters.eval$b = apply(quadrature_nodes, 1, pde.parameters$b)
        if("c" %in% names(pde.parameters) &&
           !is.null(pde.parameters[["c"]])) ## reaction
            pde.parameters.eval$c = apply(quadrature_nodes, 1, pde.parameters$c)
        if("u" %in% names(pde.parameters) &&
           !is.null(pde.parameters[["u"]])) ## forcing
            pde.parameters.eval$u = apply(quadrature_nodes, 1, pde.parameters$u)
    }

    ## cal solve on underlying module
    sol = pde$solve(observations     = observations,
                    covariates       = covariates,
                    locations        = locations, 
                    incidence_matrix = incidence_matrix,
                    pde.parameters   = pde.parameters)
}
