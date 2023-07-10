## PS: i nomi dei metodi/oggetti sono indicativi, possiamo poi cambiarli
## PS: questo codice è solo un concept, di fatto nulla è stato implementato ancora. Quando abbiamo
##     un'idea chiara e concordante si passa all'implementazione effettiva

## define mesh ...
## risolvere in maniera elegante questa parte richiede lo sviluppo del meshatore,
## per ora supponiamo di ricevere la mesh, e di gestirla con triangle attraverso le funzioni create.mesh. di fdaPDE

## define differential operator
## l'idea è quella di permettere all'utente di definire la PDE cosi come "la si scrive sulla carta". ispirato all'interfaccia
## di fenics e freeFEM
f <- FEfunction()   ## qualcosa che dia l'idea che f è una funzione scritta come espansione rispetto ad una base FEM
## l'introduzione di questo oggetto f, ci permette anche di scrivere problemi accoppiati andando a definire altre FEfunction()
## questo comunque è da vedersi...

## scrittura dell'operatore differenziale, alcuni esempi:
L <- -laplacian(f)                         ## laplaciano
L <- -div(K * grad(f)) + dot(b, grad(f))   ## diffusione - trasporto
L <- dot(b, grad(f)) + c * f               ## trasporto - reazione

## se K,b,c sono delle funzioni allora si avrà una PDE a coefficienti non constanti, ad esempio
K <- function(points) { return(matrix(...)) }
L <- -div(K * grad(f))                     ## diffusione a coefficienti non-constanti

## da vedere poi come scrivere problemi più complessi (problemi accoppiati?? se di interesse...)

## forzante, passata sempre come funzione R
u <- function(points) { return(0) } ## null force

## entry point for pde definition Lf = u on domain D (D è la mesh)
pde <- make.pde(D, L, u)
## pde è un modulo di Rcpp, nascondiamo i dettagli della sua creazione dietro una chimata make.pde, molto più di alto livello.
## no uso di operatore new, l'utente non deve sapere i dettagli del modulo che vuole instanziare, non deve richiamare i nodi
## di quadratura per valutare la forzante, i parametri della PDE per PDE con coefficienti non-costanti, etc.

## definition of dirichlet boundary conditions
dirichlet_bc <- as.matrix(rep(0., times = dim(unit_square$nodes)[1])) ## può in principio essere data anche come funzione...
pde$set_dirichlet_bc(dirichlet_bc)
## sarà poi possibile definire condizioni al bordo di natura differente (dirichlet/neumann) su porzioni di bordo differenti

## solve differential problem
pde$solve()
## access to problem solution, plot, etc.
plot(pde)
pde$solution
## ...
