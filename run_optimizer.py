from scipy.optimize import differential_evolution
from time import time
from utils import *
from optimizer_parameters import *

top = time()
print("Optimization in progress ...")
result = differential_evolution(f, bounds, maxiter=maxiter, disp=False)
print("Optimization done in {calc_time} "
      "s.\n".format(calc_time=round(time() - top, 3)))

x = result.x
alpha = 1 / (1000 * x[0])
beta = 1 / (1000 * x[1])
gamma = 1 / (1000 * x[2])

print("Alpha = {alpha}\nBeta = {beta}\nGamma = {gamma}".format(alpha=alpha,
                                                               beta=beta,
                                                               gamma=gamma))
