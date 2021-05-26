from utils import *
from simulation_parameters import *

# Transition matrix
M = calculate_transition_matrix(alpha, beta, gamma, delta_t, N)

# Initialization : January 24th 2020
numberInfected = 2
infected = np.array([0 for _ in range(N + 1)])
infected[numberInfected] = 1

# Forecast number of infected individual for April 9th 2020
# print the highest probability prevision
spread_day = 76  # number of days between January 24th 2020 & April 9th 2020
infected_forecast = np.dot(matrix_power(M, int(spread_day * 86400 / delta_t)),
                           infected)
prediction = infected_forecast.argmax()
print("According to the model, we predict that {p} people will be infected, "
      "which represents {percentage}% of the total "
      "population".format(p=prediction,
                          percentage=round(100 * prediction / N, 2)))
