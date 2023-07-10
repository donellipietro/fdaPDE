finalize <- function(env) {
   readline(prompt="save fdaPDE data to persistent storage? [y/n]: ")
}

.onAttach <- function(libname, pkgname) {
   parent <- parent.env(environment())
   reg.finalizer(parent, finalize, onexit= TRUE)
}
