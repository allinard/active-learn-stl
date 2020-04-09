Active Learning of Signal Temporal Logic Specifications
=============================================

active-learn-stl is a Python library that infers Signal Temporal Logic Specifications, using Active Learning



## Introduction

STL is a temporal logic defined on continuous signals that can be used for specifications on systems. 
It can express system properties that include time bounds and bounds on physical system parameters.
One of its advantag is its richness to specify continuous-time systems.
 It is, in the case of robotics, an attractive formalism to model, for instance, classes of desired trajectories.





## Downloading sources

You can use this API by cloning this repository:
```
$ git clone https://github.com/allinard/active-learn-stl.git
```

Dependencies:
* Python 3
	* Pulp
	* Sympy
* Gurobi MILP Solver





## STL

The module `STL.py` implements the formalism of STL Formulae.





## STL Distance

The module `STLDistance.py` implements the calculation of the distance between 2 STL Formulae.





## STL Generate Signals

The module `STLGenerateSignal.py` implements the generation of signals satisfying an STL Formula.





## STL DT Learn

The module `STLDTLearn.py` implements the learning of a decision tree for STL Inference.





## STL Active Learn

The module `STLActiveLearn.py` implements the active learning framework for the inference of STL Formula.





