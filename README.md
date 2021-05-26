# COVID19 - A Discrete Stochastic Model

The aim of this project is to implement a new discrete stochastic model in order to make predictions and to estimate some metrics such as R0.

The data have been extracted from the database [coronavirus.politologue](https://coronavirus.politologue.com). It contains information from January 24th 2020 to April 9th 2020.

## Model description

### Introduction

The model is a modified version of an already existing model first introduced in a [paper written by Hugo Falconet and Antoine Jego](http://www.math.ens.fr/enseignement/telecharger_fichier.php?fichier=1693).


Using a model based on Markov chains has several advantages. For example, such a model is **not deterministic** which allows us to take into account fluctuations regarding contamination, propagation, incubation time, etc ...

### Model

<p align="center">
  <img src="https://github.com/AdrienC21/covid-discrete-stochastic-model/blob/main/images/model-1.png" width="700">
  <img src="https://github.com/AdrienC21/covid-discrete-stochastic-model/blob/main/images/model-2.png" width="700">
  <img src="https://github.com/AdrienC21/covid-discrete-stochastic-model/blob/main/images/model-3.png" width="700">
  <img src="https://github.com/AdrienC21/covid-discrete-stochastic-model/blob/main/images/model-4.png" width="700">
  <img src="https://github.com/AdrienC21/covid-discrete-stochastic-model/blob/main/images/model-5.png" width="700">
</p>


## Installation

Clone this repository :

```bash
git clone https://github.com/AdrienC21/covid-discrete-stochastic-model.git
```
## Usage

**Run a simulation knowing the parameters (alpha, beta, gamma) :**

Edit the file simulation_parameters.py with the following information :

```python
N = 700  # total population
delta_t = 10  # time step in seconds
alpha = 8.867883874357922e-07  # death rate
beta = 2.9340382291178016e-07  # virus spread speed
gamma = 2.6457998446967257e-07  # cure rate
```

Run run_simulation.py to obtain the predicted number of infected individuals.

**Fit our model to obtain alpha, beta, gamma :**

Edit the file optimizer_parameters.py with the following information :

```python
N = 70  # total population
delta_t = 3600  # in seconds
# boundaries for  1 / 1000 * alpha, same for beta & gamma
bounds = tuple([(1., 5000.), (1., 5000.), (1., 5000.)])
maxiter = 50  # maximum number of iterations
# multiply our number of infected (more accurate predictions)
scale_factor = 500
country = "France"
normalize_pop = 1000000  # recommanded : real population number / N
```

Run run_optimizer.py to obtain alpha, beta and gamma using optimization methods.

## Documentation

### utils.py

[extract_data](https://github.com/AdrienC21/covid-discrete-stochastic-model/blob/a06d577a9fde6a02e561b8c7d606ebaffa7835ee/utils.py#L12)
```python
country = "France"
L = extract_data(country)
```
Extract the data corresponding to one country (number of infected individuals, dead people and cured ones). Sort by date.

[calculate_transition_matrix](https://github.com/AdrienC21/covid-discrete-stochastic-model/blob/a06d577a9fde6a02e561b8c7d606ebaffa7835ee/utils.py#L32)
```python
N = 700  # total population
delta_t = 10  # time step in seconds
alpha = 8.867883874357922e-07  # death rate
beta = 2.9340382291178016e-07  # virus spread speed
gamma = 2.6457998446967257e-07  # cure rate
M = calculate_transition_matrix(alpha, beta, gamma, delta_t, N)
```
Return the transition matrix of the model.

[predict](https://github.com/AdrienC21/covid-discrete-stochastic-model/blob/a06d577a9fde6a02e561b8c7d606ebaffa7835ee/utils.py#L62)
```python
x = np.array([alpha, beta, gamma])
prediction_array = predict(x)
```
Predict the number of infected people at each timestamp.

[f](https://github.com/AdrienC21/covid-discrete-stochastic-model/blob/a06d577a9fde6a02e561b8c7d606ebaffa7835ee/utils.py#L92)
```python
x = np.array([alpha, beta, gamma])
MSE = f(x)
```
Calculate the MSE (Mean Squared Error) between our predictions and the reality (Y). This is the function we want to minimize in order to obtain the best estimation for our parameters alpha, beta and gamma.
## License
[MIT](https://choosealicense.com/licenses/mit/)
