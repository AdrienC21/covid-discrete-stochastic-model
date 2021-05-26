N = 70  # total population
delta_t = 3600  # in seconds
# boundaries for  1 / 1000 * alpha, same for beta & gamma
bounds = tuple([(1., 5000.), (1., 5000.), (1., 5000.)])
maxiter = 50  # maximum number of iterations
# multiply our number of infected (more accurate predictions)
scale_factor = 500
country = "France"
normalize_pop = 1000000  # recommanded : real population number / N
