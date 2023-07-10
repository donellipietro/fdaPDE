## finite element function
fe_function <- setRefClass(
    "fe_function"
)

## take gradient of fe_function
grad <- function(x, ...) UseMethod("grad")
grad.fe_function <- function(f) {
    f_ <- list()
    class(f_) <- "fe_gradient"
    return(f_)
}

## overload product operator for fe_function gradient
`*.fe_gradient` <- function(K, f) {
    if ((!is.matrix(K)) && (!is.function(K))) {
        stop("wrong argument type")
    }
    op_ <- list(K = K, f = f)
    class(op_) <- "fe_gradient_prod"
    return(op_)
}

## laplace operator, isotropic stationary diffusion
laplace <- function(f) {
    ## issue error if f is not a finite element function
    if (!is(f, "fe_function")) {
        stop("expected argument of class fe_function")
    }
    op_ <- list(tag = "laplace", f = f)
    class(op_) <- "diff_op"
    return(op_)    
}

## divergence operator, accepting a constant or space-varying diffusion tensor
div <- function(f)  {
    if( !is(f, "fe_gradient_prod") ){
        stop("wrong argument type")
    }
    op_ <- list(tag = "divergence", param = f$K, f = f)
    class(op_) <- "diff_op"
    return(op_)
}

## dot product operator, for transport term
dot <- function(x, ...) UseMethod("dot")
dot.fe_gradient <- function(b, f) {
    op_ <- list(tag = "dot", param = b, f = f)
    class(op_) <- "diff_op"
    return(op_)    
}

## reaction term is given as the product between a constant or a function and
## a fe_function object
`*.fe_function` <- function(c, f) {
    if (!is.function(c) && !is.numeric(c)) {
        stop("wrong argument type")
    }
    op_ <- list(tag = "identity", param = c, f = f)
    class(op_) <- "diff_op"
    return(op_)

}

## composition of differential operators
`+.diff_op` <- function(op1, op2) {
    op_ <- list(
        tag = c(op1$tag, op2$tag),
        param = c(op1$param, op2$param)
    )
    class(op_) <- "diff_op"
    return(op_)
}

