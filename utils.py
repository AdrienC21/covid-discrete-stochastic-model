import csv
import pandas as pd
import numpy as np
from scipy.sparse import diags
from numpy.linalg import matrix_power
from sklearn.metrics import mean_squared_error
from optimizer_parameters import *

df = pd.read_csv(".\\covid19_chiffre.csv", delimiter=";")


def extract_data(country):  # country="France"
    """Extract the data corresponding to one country

    Args:
        country (str): on which country we want to extract the data

    Returns:
        list: List of number of infected individuals, dead people and the
              cured ones. Sort by date.
    """
    L = []
    df2 = df[df["Pays"] == country]
    df2 = df2[1:]  # last line twice
    for id, (Date, Pays, Infections, Deces, Guerisons,
             _, _, _) in df2.iterrows():
        L.append([Infections, Deces, Guerisons])
    L.reverse()
    return L


def calculate_transition_matrix(alpha, beta, gamma, delta_t, N):
    """Return the transition matrix of the model.

    Args:
        alpha (float): death rate
        beta (float): spread speed
        gamma (float): cure rate
        delta_t (float): time step
        N (int): total population

    Returns:
        array: Transition matrix
    """

    M = diags([0, 0, 0], [-1, 0, 1], shape=(N+1, N+1)).toarray()

    M[0, 0] = 1
    M[0, 1] = 0
    for i in range(1, N):
        bi = delta_t * beta * i * (N - i) / N
        di = (alpha + gamma) * i * delta_t
        M[i, i] = 1 - (bi + di)
        M[i, i + 1] = bi
        M[i, i - 1] = di
    M[N, N - 1] = (alpha + gamma) * N * delta_t
    M[N, N] = 1 - (alpha + gamma) * N * delta_t

    return M


def predict(x):
    """Predict the number of infected people at each timestamp.

    Args:
        x (array): Contains alpha, beta and gamma

    Returns:
        array: Number of infected people at each timestamp.
    """
    global alpha, beta, gamma, delta_t, N, X, Y

    inv_alpha, inv_beta, inv_gamma = x[0], x[1], x[2]
    alpha = 1 / (inv_alpha * 1000)
    beta = 1 / (inv_beta * 1000)
    gamma = 1 / (inv_gamma * 1000)

    M = calculate_transition_matrix(alpha, beta, gamma, delta_t, N)

    A = matrix_power(M, int(86400 / delta_t))

    Y_pred = []
    for a in X:
        infected = np.array([0 for _ in range(N + 1)])
        infected[int(a)] = 1
        infected = np.dot(A, infected)
        Y_pred.append(infected.argmax())

    return np.array(Y_pred)


def f(x):
    """Calculate the MSE (Mean Squared Error) between our predictions
    and the reality (Y).

    Args:
        x (array): Contains alpha, beta and gamma

    Returns:
        float: Mean Squared Error
    """
    global Y

    return mean_squared_error(Y, predict(x))


L = [int(scale_factor * infected / normalize_pop)
     for [infected, _, _] in extract_data(country)]
X = L[:-1]
Y = [L[i+1] for i in range(len(L)-1)]
