import csv
import pandas as pd
import numpy as np

df = pd.read_csv(".\\covid19_chiffre.csv", delimiter=";")

def liste_des_pays():
    L = df["Pays"].to_numpy()
    for elt in L:
        print(elt)

def extract_data(country): #country="France"
    L = []
    df2 = df[df["Pays"]==country]
    df2 = df2[1:]
    for id, (Date,Pays,Infections,Deces,Guerisons,_,_,_) in df2.iterrows():
        L.append([Infections,Deces,Guerisons])
    L.reverse()
    return L

##

"""
Test : 700 habitants
"""

from scipy.sparse import diags
from numpy.linalg import matrix_power

#Paramètres
N = 700
delta_t = 10 #10s
alpha = 1/2000000
gamma = 1/3000000
beta = 1/20000000000

#Matrice de transition
M = diags([0, 0, 0], [-1, 0, 1], shape=(N+1, N+1)).toarray()

M[0,0] = 1
M[0,1] = 0
for i in range(1,N):
    bi = delta_t*beta*i*(N-i)/N
    di = (alpha+gamma)*i*delta_t
    M[i,i] = 1-(bi+di)
    M[i,i+1] = bi
    M[i,i-1] = di
M[N,N-1] = (alpha+gamma)*N*delta_t
M[N,N] = 1-(alpha+gamma)*N*delta_t

#Initialisation : 24 janvier 2020
nbInfectés = 2
I = np.array([0 for _ in range(N+1)])
I[nbInfectés] = 1

#Probabilités pour le 9 avril 2020 : on affiche le nombre d'infectés avec la probabilité maximale
I = np.dot(matrix_power(M, 76*8640), I)
print(I.argmax())

##Fit sur la France pour trouver alpha, beta, gamma

"""
Problème :
    Compléxité énorme qui ne peut pas vraiment être réduite en introduisant des facteurs pour réduire les chiffres car les matrices prennent en compte des entiers

Diminuer la complexité :
- réduire delta_t
- réduire N
Dans les deux cas, le fit perd son sens et n'est plus possible car la valeur de sortie de notre fonction est trop faible.

Choix : augmenter le nombre d'infecté d'un facteur 500
"""

from lmfit import Model

N = 700 #facteur 100000 pour population total de 70 000 000 (matrice puissance 8640 (24h/10 (delta_t)) faisable)
delta_t = 10 #10s

L = [500*infectés/100000 for [infectés,déces,guérisons] in extract_data("France")]
x = L[:-1]
y = [L[i+1] for i in range(76)]

#On travaille sur les inverses de nos paramètres pour fiter
def f(x,inv_alpha,inv_beta,inv_gamma):
    alpha =  1 / inv_alpha
    beta = 1 / inv_beta
    gamma = 1 / inv_gamma


    M = diags([0, 0, 0], [-1, 0, 1], shape=(N+1, N+1)).toarray()

    M[0,0] = 1
    M[0,1] = 0
    for i in range(1,N):
        bi = delta_t*beta*i*(N-i)/N
        di = (alpha+gamma)*i*delta_t
        M[i,i] = 1-(bi+di)
        M[i,i+1] = bi
        M[i,i-1] = di
    M[N,N-1] = (alpha+gamma)*N*delta_t
    M[N,N] = 1-(alpha+gamma)*N*delta_t

    A = matrix_power(M,8640)
    if type(x) == np.ndarray:
        res = []
        for a in x:
            I = np.array([0 for _ in range(N+1)])
            I[int(a)] = 1
            I = np.dot(A, I)
            res.append(I.argmax())
        return np.array(res)
    else:
        I = np.array([0 for _ in range(N+1)])
        I[int(x)] = 1
        I = np.dot(A, I)
        return I.argmax()

#Model & fit
gmodel = Model(f)
result = gmodel.fit(y, x=x, inv_alpha = 160000, inv_beta = 24000, inv_gamma = 160000)

print(result.fit_report())

##Tentative de BruteForce pour trouver les paramètres (regroupe alpha et gamma pour réduire temps de calcul)

from lmfit import Minimizer, Parameters, fit_report

L = [500*infectés/100000 for [infectés,déces,guérisons] in extract_data("France")]
x = L[:-1]
y = [L[i+1] for i in range(76)]

params = Parameters()
params.add_many(
        ('N', 700, False),
        ('delta_t', 10, False),
        ('inv_alpha_gamma', 50000, True),
        ('inv_beta', 20000, True))
#N facteur 100000 pour population total de 70 000 000 (matrice puissance 8640 (24h/10 (delta_t)) faisable)
#delta_t 10s

params['inv_alpha_gamma'].set(min=100000, max=200000, brute_step=10000)
params['inv_beta'].set(min=5000, max=25000, brute_step=1000)

def f(p):
    par = p.valuesdict()

    alpha =  1 / par['inv_alpha_gamma']
    beta = 1 / par['inv_beta']
    gamma = 1 / par['inv_alpha_gamma']

    M = diags([0, 0, 0], [-1, 0, 1], shape=(par['N']+1, par['N']+1)).toarray()

    M[0,0] = 1
    M[0,1] = 0
    for i in range(1,par['N']):
        bi = par['delta_t']*beta*i*(par['N']-i)/par['N']
        di = (alpha+gamma)*i*par['delta_t']
        M[i,i] = 1-(bi+di)
        M[i,i+1] = bi
        M[i,i-1] = di
    M[par['N'],par['N']-1] = (alpha+gamma)*par['N']*par['delta_t']
    M[par['N'],par['N']] = 1-(alpha+gamma)*par['N']*par['delta_t']

    A = matrix_power(M,8640)
    res = []
    for a in x:
        I = np.array([0 for _ in range(par['N']+1)])
        I[int(a)] = 1
        I = np.dot(A, I)
        res.append(I.argmax())
    res = np.array(res)-np.array(y)

    quad = 0
    for elt in res:
        quad = quad+elt**2
    return np.sqrt(quad)


fitter = Minimizer(f, params)
result = fitter.minimize(method='brute')

print(result.brute_x0)

#[80000. 24000.] (alpha*gamma)/(alpha+gamma), beta