# COVID19 - A Discrete Stochastic Model

The aim of this project is to implement a new discrete stochastic model in order to make predictions and to estimate some metrics such as R0.

The data have been extracted from the database [coronavirus.politologue](https://coronavirus.politologue.com). It contains information from January 24th 2020 to April 9th 2020.

## Model description

### Introduction

The model is a modified version of an already existing model first introduced in a [paper written by Hugo Falconet and Antoine Jego](http://www.math.ens.fr/enseignement/telecharger_fichier.php?fichier=1693).


Using a model based on Markov chains has several advantages. For example, such a model is **not determinist** which allows us to take into account fluctuations regarding contamination, propagation, incubation time, etc ...

### Model

![hustlin_erd](pdf/model.pdf)

## Installation

Clone this repository :

```bash
git clone https://github.com/AdrienC21/covid-discrete-stochastic-model.git
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
