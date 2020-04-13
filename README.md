Active Learning of Signal Temporal Logic Specifications
=============================================

active-learn-stl is a Python library that infers Signal Temporal Logic Specifications, using Active Learning



## Introduction

STL is a temporal logic defined on continuous signals that can be used for specifications on systems. 
It can express system properties that include time bounds and bounds on physical system parameters.
One of its advantage is its richness to specify continuous-time systems.
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
It supports several boolean (Conjunction, Disjunction, Negation) and temporal operators (Always, Eventually).


### True and False boolean constants

```
t = STLFormula.TrueF()
f = STLFormula.FalseF()
```


### Predicate

```
x_gt3 = STLFormula.Predicate(dimension,operator,mu,pi_index_signal)
```
is an STL Formula, where the constructor takes 4 arguments:
* dimension: string/name of the dimension (ex: 'x')
* operator: operator (operatorclass.geq, operatorclass.lt, operatorclass.leq, operatorclass.gt)
* mu: float mu (ex: 3)
* pi_index_signal: in the signal, which index corresponds to the predicate's dimension (ex: 0)


### Conjunction and Disjunction

```
c = STLFormula.Conjunction(phi1,phi2)
d = STLFormula.Disjunction(phi1,phi2)
```
are STL Formulae respectively representing the Conjunction and Disjunction of 2 STL Formulae phi1 and ph2.


### Negation

```
n = STLFormula.Negation(phi)
```
is an STL Formula representing the negation of an STL Formula phi.


### Always and Eventually

```
a = STLFormula.Always(phi,t1,t2)
e = STLFormula.Eventually(phi,t1,t2)
```
are STL Formulae respectively representing the Always and Eventually of an STL Formulae phi. They both takes 3 arguments:
* formula: a formula phi
* t1: lower time interval bound
* t2: upper time interval bound


### Robustness

All STL Formulae contain 1 function to compute the robustness of a signal given the STL Formula.

```
x_gt3 = STLFormula.Predicate('x',operatorclass.gt,3,0)
a = STLFormula.Always(x_gt3,0,5)
a.robustness([[3.1],[3.3],[3.2],[3.0],[2.9],[3.1],[3.5],[3.1],[2.2]],0)
-0.1
```



## STL Distance

The module `STLDistance.py` implements the calculation of the distance between 2 STL Formulae.

Follows the definitions of Madsen et al., "Metrics  for  signal temporal logic formulae," in 2018 IEEE Conference on Decision and Control (CDC). pp. 1542–1547

```
rand_area = [0,1]

x_ge02 = STLFormula.Predicate('x',operatorclass.ge,0.2,INDEX_X)
x_lt04 = STLFormula.Predicate('x',operatorclass.le,0.4,INDEX_X)
x_lt044 = STLFormula.Predicate('x',operatorclass.le,0.44,INDEX_X)
phi1 = STLFormula.Always(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt04),0,20)
phi2 = STLFormula.Always(STLFormula.Conjunction(predicate_x_ge02,predicate_x_lt044),0,20)

pompeiu_hausdorff_distance(phi1,phi2,rand_area)
0.04
```
where `pompeiu_hausdorff_distance` takes as input:
* phi1: an STL Formula
* phi2: an STL Formula
* rand_area: the domain on which signals are generated. rand_area = [lb,ub] where lb is the lower bound and ub the upper bound of the domain.



## STL Generate Signals

The module `STLGenerateSignal.py` implements the generation of signals satisfying an STL Formula.





## STL DT Learn

The module `STLDTLearn.py` implements the learning of a decision tree for STL Inference.





## STL Active Learn

The module `STLActiveLearn.py` implements the active learning framework for the inference of STL Formula.





