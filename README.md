# Quantum alternating operator ansatz and QAOA for the travelling salesperson problem


This repo is a qiskit implementation of the [quantum alternating operator ansatz](https://arxiv.org/abs/1709.03489) and the [Quantum approximate optimization algorithm](https://arxiv.org/abs/1411.4028) for the well known travelling salesperson problem. It was inspired by the work of [Radzihovsky, Murphy and Swofford](https://github.com/murphyjm/cs269q_radzihovsky_murphy_swofford).


## Installation
To clone this repo use:

`git clone git@github.com:simondobers/quantum_alternating_operator.git`

Install required packages via

`pip install -r requirements.txt`

or

`conda env create --file=environment.yml`

if you have anaconda installed.

## Codebase
`alternating_operator.py` implements all the alternating operator ansatz algorithm

`qaoa.py` implemts the constrained QAOA solution to solve the TSP problem 

Helper functionalites, such as creating the problem graph or plotting purposes are implemented in `helper.py`

A classical (brute-force) solver for the TSP problem is implemented in `classical.py`

`anim.py` is used for creating animations of how the statevectors evolve during optimization 

## References
[[1]](https://doi.org/10.48550/arxiv.1411.4028) 
Farhi, Edward and Goldstone, Jeffrey and Gutmann, Sam (2014). 
A Quantum Approximate Optimization Algorithm. 


[[2]](https://doi.org/10.48550/arxiv.1411.4028) 
Hadfield et. al. (2019). 
From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz. 